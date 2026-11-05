// Bot runtime: drives one player of the wasm Sim, either with a scripted Rust bot or with an ONNX
// policy. Everything about the observation and the action format comes from the Rust code compiled
// into sim-wasm (bots::observe, bots::Action::to_input, bots::Scripted): nothing is re-implemented here.
//
// Policy timing matches training: one decision per 4 sim steps (30 Hz at 120 Hz), the action held
// in between, the deterministic outputs (argmax move, mean stick direction, dash/lift prob > 0.5).
// onnxruntime-web's run() is asynchronous, so a decision's action applies from the first sim step
// after it resolves (in practice the next frame); a decision is skipped if the previous one is
// still running (counted in stats.skipped).

import * as ort from "onnxruntime-web/wasm";
import type { Sim } from "../../wasm-pkg/sim_wasm.js";
import type { PlayerInput } from "../types";
import { handicapFor, type Handicap } from "./skill";

export const DECISION_STEPS = 4;
const IDLE: PlayerInput = { move_x: 0, move_y: 0, stick_target_angle: 0, dash: false, lift: false };

let ortReady = false;
function setupOrt(): void {
  if (ortReady) return;
  // onnxruntime-web (pinned in package.json) is bundled by Vite; its wasm runtime is emitted into
  // dist/assets/ and loaded from a relative URL (no CDN). Single-threaded: no SharedArrayBuffer, so
  // no cross-origin isolation headers are needed (GitHub Pages can't set them).
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ortReady = true;
}

/** Load an ONNX policy exported by train/export_onnx.py; checks its sidecar against the wasm observation. */
export async function loadPolicy(url: string, obsDim: number, obsVersion: number): Promise<ort.InferenceSession> {
  setupOrt();
  const side = await fetch(url.replace(/\.onnx$/, ".json"));
  if (side.ok) {
    const s = await side.json();
    if (s.obs_dim !== obsDim || s.obs_version !== obsVersion || (s.act_version ?? 1) !== 2) {
      throw new Error(`policy layout obs v${s.obs_version}/${s.obs_dim} act v${s.act_version ?? 1} does not match the game (obs v${obsVersion}/${obsDim}, act v2)`);
    }
  }
  const buf = await fetch(url);
  if (!buf.ok) throw new Error(`HTTP ${buf.status} for ${url}`);
  return ort.InferenceSession.create(new Uint8Array(await buf.arrayBuffer()), { executionProviders: ["wasm"] });
}

/** Small seeded RNG (mulberry32) for the handicaps. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface BotStats {
  decisions: number;
  skipped: number;
  inferenceMs: number[];
  mistakes: number;
}

export abstract class BotDriver {
  readonly hc: Handicap;
  protected rand: () => number;
  protected step = 0;
  private mistakeLeft = 0;
  private mistakeDir = 0;
  private noise = 0;
  stats: BotStats = { decisions: 0, skipped: 0, inferenceMs: [], mistakes: 0 };

  constructor(protected sim: Sim, readonly player: number, skill: number, seed = 1) {
    this.hc = handicapFor(skill);
    this.rand = rng(seed);
  }

  abstract readonly label: string;
  /** This sim step's input for the bot (call once per step, before sim.step). */
  abstract input(): PlayerInput;
  reset(): void {
    this.step = 0;
    this.mistakeLeft = 0;
  }

  /** Gaussian noise for the stick, redrawn once per decision. */
  protected stickNoise(): number {
    if (this.step % DECISION_STEPS === 0 && this.hc.stickNoise > 0) {
      const u = Math.max(1e-12, this.rand()), v = this.rand();
      this.noise = this.hc.stickNoise * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    return this.noise;
  }

  /** Mistake handicap: occasionally skate a random direction (canonical move 1..8) with the stick still. Returns the move index or -1. */
  protected mistake(): number {
    if (this.mistakeLeft > 0) {
      this.mistakeLeft--;
      return this.mistakeDir;
    }
    if (this.hc.mistakeRate > 0 && this.rand() < this.hc.mistakeRate / 120) {
      this.mistakeLeft = this.hc.mistakeSteps;
      this.mistakeDir = 1 + Math.floor(this.rand() * 8);
      this.stats.mistakes++;
      return this.mistakeDir;
    }
    return -1;
  }
}

/** A scripted Rust bot (bots::Scripted through sim-wasm). Delay is applied to its output (the bot
 * itself reads the live state inside Rust): the input it computes is used `delaySteps` later. */
export class ScriptedDriver extends BotDriver {
  private queue: PlayerInput[] = [];
  constructor(sim: Sim, player: number, readonly kind: string, skill: number, seed = 1) {
    super(sim, player, skill, seed);
    sim.set_scripted(player, kind);
  }
  get label(): string {
    return `scripted ${this.kind}`;
  }
  input(): PlayerInput {
    const inp = JSON.parse(this.sim.scripted_input(this.player)) as PlayerInput;
    inp.stick_target_angle += this.stickNoise();
    const m = this.mistake();
    if (m >= 0) {
      const a = JSON.parse(this.sim.action_input(this.player, m, 0, false, false)) as PlayerInput;
      inp.move_x = a.move_x;
      inp.move_y = a.move_y;
      inp.dash = false;
      inp.lift = false;
    }
    this.step++;
    this.queue.push(inp);
    const out = this.queue.length > this.hc.delaySteps ? this.queue.shift()! : { ...IDLE };
    return out;
  }
  reset(): void {
    super.reset();
    this.queue = [];
  }
}

/** An ONNX policy: obs from sim.observe (with the style), 30 decisions/s, deterministic heads. */
export class PolicyDriver extends BotDriver {
  private obsHist: Float32Array[] = [];
  private pending = false;
  private held: PlayerInput = { ...IDLE };
  private action = { move: 0, angle: 0, dash: false, lift: false };
  readonly style: Float32Array;

  constructor(sim: Sim, player: number, private session: ort.InferenceSession, readonly name: string, style: number[], skill: number, seed = 1) {
    super(sim, player, skill, seed);
    this.style = new Float32Array(style);
  }
  get label(): string {
    return `policy ${this.name}`;
  }
  input(): PlayerInput {
    // Perception delay: the policy decides on the observation from `delaySteps` ago.
    this.obsHist.push(this.sim.observe(this.player, this.style));
    if (this.obsHist.length > this.hc.delaySteps + 1) this.obsHist.shift();
    if (this.step % DECISION_STEPS === 0) {
      if (this.pending) this.stats.skipped++;
      else this.decide(this.obsHist[0]);
    }
    const m = this.mistake();
    let out = this.held;
    if (m >= 0) out = JSON.parse(this.sim.action_input(this.player, m, this.action.angle, false, false));
    this.step++;
    return out;
  }
  private decide(obs: Float32Array): void {
    this.pending = true;
    const t0 = performance.now();
    const noise = this.stickNoise();
    this.session
      .run({ obs: new ort.Tensor("float32", obs, [1, obs.length]) })
      .then((r) => {
        this.stats.inferenceMs.push(performance.now() - t0);
        this.stats.decisions++;
        const sd = r.stick_dir.data as Float32Array;
        this.action = {
          move: Math.round((r.move.data as Float32Array)[0]),
          angle: Math.atan2(sd[1], sd[0]) + noise,
          dash: (r.dash_prob.data as Float32Array)[0] > 0.5,
          lift: (r.lift_prob.data as Float32Array)[0] > 0.5,
        };
        // bots::Action::to_input (team rotation) in Rust.
        this.held = JSON.parse(this.sim.action_input(this.player, this.action.move, this.action.angle, this.action.dash, this.action.lift));
      })
      .catch((e) => console.error("policy run failed", e))
      .finally(() => {
        this.pending = false;
      });
  }
  reset(): void {
    super.reset();
    this.obsHist = [];
    this.held = { ...IDLE };
  }
}
