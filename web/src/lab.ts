// Lab core: owns the wasm Sim, the param file, recording, replay and practice tools.
// No DOM and no rendering here. Wall-clock time only enters through `realDt` arguments from the caller,
// and practice tools touch the state only outside recording/replay, so recorded runs replay exactly.

import type { Sim } from "../wasm-pkg/sim_wasm.js";
import { retimeMatch } from "./panel";
import { flatParams, type InputEvent, type P, type ParamFile, type PlayerInput, type Sample, type Scenario, type State } from "./types";

export type Mode = "play" | "record" | "replay";

const IDLE = (angle: number): PlayerInput => ({ move_x: 0, move_y: 0, stick_target_angle: angle, dash: false, lift: false });

/** Puck spawn height rises this fast while the spawn key is held (approximate tool). */
export const SPAWN_RAISE_RATE = 1.5; // m/s
export const SPAWN_MAX_Z = 6.0; // m
/** Q-call: puck arrives at the forehand point with roughly this speed (approximate tool). */
export const CALL_ARRIVE_SPEED = 3.0; // m/s
export const CALL_MAX_SPEED = 20.0; // m/s

/** Matches calib::write_csv: header t,x,y,z,stick_angle and shortest round-trip numbers. */
export function toCsv(samples: Sample[]): string {
  const withStick = samples.some((s) => s.stick_angle !== undefined);
  let out = withStick ? "t,x,y,z,stick_angle\n" : "t,x,y,z\n";
  for (const s of samples) {
    out += `${s.t},${s.x},${s.y},${s.z}`;
    if (withStick) out += `,${s.stick_angle ?? ""}`;
    out += "\n";
  }
  return out;
}

/** Matches calib::parse_csv (header-indexed columns, '#' comments skipped). */
export function parseCsv(text: string): Sample[] {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
  const header = lines.shift()?.split(",").map((h) => h.trim()) ?? [];
  const col = (n: string) => header.indexOf(n);
  const [it, ix, iy, iz, ia] = ["t", "x", "y", "z", "stick_angle"].map(col);
  if (it < 0 || ix < 0 || iy < 0 || iz < 0) throw new Error("csv needs t,x,y,z columns");
  return lines.map((l) => {
    const f = l.split(",");
    const s: Sample = { t: +f[it], x: +f[ix], y: +f[iy], z: +f[iz] };
    if (ia >= 0 && f[ia] !== undefined && f[ia].trim() !== "") s.stick_angle = +f[ia];
    return s;
  });
}

function sameInput(a: PlayerInput, b: PlayerInput): boolean {
  return a.move_x === b.move_x && a.move_y === b.move_y && a.stick_target_angle === b.stick_target_angle && a.dash === b.dash && a.lift === b.lift && !!a.stick_up === !!b.stick_up && !!a.brake === !!b.brake; // w4/w5b: raised stick and brake are logged too
}

/** `endHash`: sim state hash after the last recorded / replayed step (determinism check). */
export interface Recording { scenario: Scenario; samples: Sample[]; endHash: string }

export class Lab {
  sim: Sim;
  file: ParamFile;
  p: P;
  state: State;
  mode: Mode = "play";

  /** Inputs for every player, used on the next step (player 0 = human). */
  inputs: PlayerInput[];

  // Recording / replay.
  private recStart: State | null = null;
  private recEvents: InputEvent[] = [];
  private recSamples: Sample[] = [];
  private recLast: PlayerInput[] = [];
  private recSteps = 0;
  private replayScenario: Scenario | null = null;
  private replayHeld: PlayerInput[] = [];
  private replayIdx = 0;
  lastRecording: Recording | null = null;
  lastReplay: Recording | null = null;
  onReplayDone: (() => void) | null = null;

  // Practice tools (approximate).
  spawn = { x: 0, y: 0, z: 0 };
  spawnHeld = false;
  target: { x: number; y: number } | null = null;
  targetBest = Infinity;

  /** Per-step observers (verification traces, UI). */
  stepListeners: ((lab: Lab) => void)[] = [];

  constructor(SimCtor: typeof Sim, file: ParamFile, nPlayers = 2) {
    this.file = structuredClone(file);
    this.p = flatParams(this.file);
    this.sim = new SimCtor(JSON.stringify(this.file), nPlayers);
    this.state = JSON.parse(this.sim.get_state());
    this.inputs = this.state.players.map((pl) => IDLE(pl.facing));
  }

  // ---------- params ----------

  setParamFile(file: ParamFile): void {
    this.sim.set_params(JSON.stringify(file));
    const was = this.p;
    this.file = structuredClone(file);
    this.p = flatParams(this.file);
    const s = structuredClone(this.state);
    if (this.mode === "play" && retimeMatch(s.match_state, was, this.p)) this.writeState(s);
  }

  setParam(name: string, value: number): void {
    const f = structuredClone(this.file);
    f.params[name].value = value;
    this.setParamFile(f);
  }

  // ---------- match mode (rules live in the sim: sim/src/rules.rs) ----------

  /** Match rules are on when the state carries a match_state (free play otherwise). */
  get inMatch(): boolean {
    return !!this.state.match_state;
  }

  /** Start a fresh match (spawn positions, 0-0, full clock, countdown). Only in play mode, not while recording/replaying. */
  startMatch(): boolean {
    if (this.mode !== "play") return false;
    this.spawnHeld = false;
    this.sim.start_match(this.state.players.length);
    this.state = JSON.parse(this.sim.get_state());
    return true;
  }

  /** Back to free play: same positions, rules off, score kept. Only in play mode. */
  endMatch(): boolean {
    if (this.mode !== "play") return false;
    const s = structuredClone(this.state);
    s.match_state = null;
    this.writeState(s);
    return true;
  }

  /** Enter after game over: a new match. */
  rematch(): boolean {
    return this.state.match_state?.phase === "finished" && this.startMatch();
  }

  // ---------- stepping ----------

  private writeState(s: State): void {
    this.sim.set_state(JSON.stringify(s));
    this.state = s;
  }

  /** Advance exactly one sim step. `realDt` (wall-clock seconds) only drives practice-tool animation. */
  step(realDt = 0): void {
    if (this.mode === "replay") {
      const sc = this.replayScenario!;
      const k = this.recSteps;
      while (this.replayIdx < sc.inputs.length && sc.inputs[this.replayIdx].step <= k) {
        const e = sc.inputs[this.replayIdx++];
        if (e.player < this.replayHeld.length) this.replayHeld[e.player] = e.input;
      }
      this.sim.step(JSON.stringify(this.replayHeld));
    } else {
      if (this.mode === "play" && this.spawnHeld) {
        this.spawn.z = Math.min(SPAWN_MAX_Z, this.spawn.z + SPAWN_RAISE_RATE * realDt);
        this.placePuckAtSpawn();
      }
      if (this.mode === "record") {
        this.inputs.forEach((inp, i) => {
          if (!this.recLast[i] || !sameInput(this.recLast[i], inp)) {
            this.recEvents.push({ step: this.recSteps, player: i, input: { ...inp } });
            this.recLast[i] = { ...inp };
          }
        });
      }
      this.sim.step(JSON.stringify(this.inputs));
    }
    this.state = JSON.parse(this.sim.get_state());
    if (this.mode === "play" && this.spawnHeld) this.placePuckAtSpawn(); // keep it pinned while held

    if (this.mode !== "play") {
      this.recSteps++;
      this.recSamples.push(this.sample(this.recSteps * this.p.timestep));
      if (this.mode === "replay" && this.recSteps >= this.replayScenario!.steps) this.finishReplay();
    }
    if (this.target) {
      const d = Math.hypot(this.state.puck.x - this.target.x, this.state.puck.y - this.target.y);
      this.targetBest = Math.min(this.targetBest, d);
    }
    for (const f of this.stepListeners) f(this);
  }

  private sample(t: number): Sample {
    const pk = this.state.puck;
    return { t, x: pk.x, y: pk.y, z: pk.z, stick_angle: this.state.players[0]?.stick_angle };
  }

  // ---------- record / replay (calib formats) ----------

  startRecording(): void {
    if (this.mode !== "play") return;
    this.spawnHeld = false;
    this.recStart = JSON.parse(this.sim.get_state());
    this.recEvents = [];
    this.recLast = [];
    this.recSteps = 0;
    this.recSamples = [this.sample(0)];
    this.mode = "record";
  }

  stopRecording(): Recording | null {
    if (this.mode !== "record") return null;
    this.mode = "play";
    const scenario: Scenario = {
      name: `lab_${this.recStart!.tick}`,
      description: "Recorded in the feel lab",
      steps: this.recSteps,
      initial_state: this.recStart!,
      inputs: this.recEvents,
    };
    this.lastRecording = { scenario, samples: this.recSamples, endHash: this.sim.state_hash() };
    return this.lastRecording;
  }

  startReplay(sc: Scenario): void {
    if (this.mode === "record") this.stopRecording();
    this.sim.reset(JSON.stringify(sc));
    this.state = JSON.parse(this.sim.get_state());
    this.replayScenario = sc;
    this.replayHeld = this.state.players.map(() => IDLE(0));
    this.replayIdx = 0;
    this.recSteps = 0;
    this.recSamples = [this.sample(0)];
    this.mode = sc.steps > 0 ? "replay" : "play";
  }

  private finishReplay(): void {
    this.mode = "play";
    this.lastReplay = { scenario: this.replayScenario!, samples: this.recSamples, endHash: this.sim.state_hash() };
    // Keep the human's stick where the replay left it so play resumes smoothly.
    this.onReplayDone?.();
  }

  get progress(): { steps: number; total: number | null } {
    return { steps: this.recSteps, total: this.mode === "replay" ? this.replayScenario!.steps : null };
  }

  // ---------- practice tools (APPROXIMATE, web layer only; disabled while recording/replaying and in match mode) ----------

  get toolsEnabled(): boolean {
    return this.mode === "play" && !this.inMatch;
  }

  private placePuckAtSpawn(): void {
    const s = structuredClone(this.state);
    s.puck = { ...s.puck, x: this.spawn.x, y: this.spawn.y, z: this.spawn.z, vx: 0, vy: 0, vz: 0, in_net: null };
    this.writeState(s);
  }

  /** F down: spawn at cursor, height starts at 0 and rises while held. */
  spawnPress(x: number, y: number): void {
    if (!this.toolsEnabled) return;
    this.spawn = { x, y, z: 0 };
    this.spawnHeld = true;
    this.placePuckAtSpawn();
  }

  /** F up: release the puck from the spawn (it falls under gravity if raised). */
  spawnRelease(): void {
    this.spawnHeld = false;
  }

  /** R: puck back to spawn at rest. */
  resetPuck(): void {
    if (!this.toolsEnabled) return;
    this.placePuckAtSpawn();
    this.targetBest = Infinity;
  }

  /** G: target marker at cursor. */
  setTarget(x: number, y: number): void {
    if (this.inMatch) return;
    this.target = { x, y };
    this.targetBest = Infinity;
  }

  /** Forehand point: near the blade, on the CCW side of the stick (handedness unverified). */
  forehandPoint(): { x: number; y: number } {
    const pl = this.state.players[0];
    const a = pl.stick_angle;
    const along = this.p.stick_length * 0.8;
    const side = this.p.puck_radius + this.p.stick_thickness + 0.25;
    return {
      x: pl.x + Math.cos(a) * along - Math.sin(a) * side,
      y: pl.y + Math.sin(a) * along + Math.cos(a) * side,
    };
  }

  /** Q: slide the puck toward the forehand point, arriving at about CALL_ARRIVE_SPEED. */
  callPuck(): void {
    if (!this.toolsEnabled) return;
    const f = this.forehandPoint();
    const s = structuredClone(this.state);
    const dx = f.x - s.puck.x;
    const dy = f.y - s.puck.y;
    const d = Math.hypot(dx, dy);
    if (d < 1e-6) return;
    const v = Math.min(CALL_MAX_SPEED, Math.sqrt(CALL_ARRIVE_SPEED ** 2 + 2 * this.p.ice_friction_decel * d));
    s.puck = { ...s.puck, z: 0, vz: 0, vx: (dx / d) * v, vy: (dy / d) * v };
    this.writeState(s);
  }

  /** Overwrite the state (verification hooks, loaded drills). Only outside record/replay. Not a practice
   * tool, so it also works in match mode (verify.mjs uses it to put pucks on goal). */
  setState(s: State): void {
    if (this.mode !== "play") return;
    this.writeState(structuredClone(s));
  }

  hash(): string {
    return this.sim.state_hash();
  }
}
