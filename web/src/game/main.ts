// The game: menu → match vs a bot (sim match rules, human = BLUE with WASD + mouse) → game over.
// The feel lab lives at lab.html (Practice). Wall-clock reads live only here.

import init, { Sim } from "../../wasm-pkg/sim_wasm.js";
import paramsFile from "../../../params.json";
import { actionFor, BINDINGS } from "../bindings";
import { FixedStepLoop } from "../loop";
import { draw, drawMatchHud, fitView, formatClock, toWorld, TEAM_NAMES, type View } from "../render";
import { flatParams, type ParamFile, type PlayerInput, type State } from "../types";
import { BotDriver, loadPolicy, PolicyDriver, ScriptedDriver } from "./botRuntime";
import { BOTS_DIR, loadProfiles, type Profile } from "./profiles";
import { describeHandicap } from "./skill";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const qs = new URLSearchParams(location.search);

await init();
const FILE = paramsFile as ParamFile;
const P = flatParams(FILE);
const sim = new Sim(JSON.stringify(FILE), 2);
let state: State = JSON.parse(sim.get_state());

// ---------- screens ----------
type Screen = "menu" | "controls" | "loading" | "match";
let screen: Screen = "menu";
const SCREENS = ["menu", "controls", "loading", "gameover"] as const;
function show(s: Screen): void {
  screen = s;
  for (const id of SCREENS) $(id).hidden = id !== s;
  if (s === "menu") $("play").focus();
  if (s === "controls") $("controls-back").focus();
}
function banner(msg: string | null): void {
  $("banner").hidden = !msg;
  $("banner").textContent = msg ?? "";
}

// ---------- menu ----------
const { profiles, errors } = await loadProfiles(BOTS_DIR);
const sel = $<HTMLSelectElement>("opponent");
sel.innerHTML = profiles.map((p) => `<option value="${p.id}">${p.name}${p.smoke_test ? " (smoke test, weak)" : ""}</option>`).join("");
const skill = $<HTMLInputElement>("skill");
function refreshMenu(): void {
  const p = profiles.find((x) => x.id === sel.value)!;
  $("opponent-desc").textContent = p.description + (p.policy ? ` · policy ${p.policy}` : ` · scripted ${p.scripted}`);
  $("skill-value").textContent = skill.value;
  $("skill-desc").textContent = describeHandicap(+skill.value) + " (uncalibrated)";
}
sel.addEventListener("change", () => {
  skill.value = String(profiles.find((x) => x.id === sel.value)!.skill);
  refreshMenu();
});
skill.addEventListener("input", refreshMenu);
refreshMenu();
if (errors.length) $("menu-status").textContent = "Some bots could not be listed: " + errors.join("; ");
$("play").addEventListener("click", () => void startMatch(sel.value, +skill.value));
$("practice").addEventListener("click", () => (location.href = "lab.html"));
$("controls-btn").addEventListener("click", () => show("controls"));
$("controls-back").addEventListener("click", () => show("menu"));
$("controls-list").innerHTML = [
  ["W A S D", "skate (8 directions, screen-relative)"],
  ["Mouse", "stick direction (the stick turns toward the cursor)"],
  ["Shift", "dash (needs a move key) — UNVERIFIED binding"],
  ["Space", "lift — UNVERIFIED binding"],
  ["Enter", "rematch after game over"],
  ["Esc", "leave the match / back to the menu"],
].map(([k, d]) => `<li><b>${k}</b> — ${d}</li>`).join("");
$("rematch").addEventListener("click", () => void startMatch(current!.id, currentSkill));
$("to-menu").addEventListener("click", () => toMenu());

// Arrow keys move focus through the focusable items of the visible screen (plus Tab, as usual).
function focusables(): HTMLElement[] {
  const s = SCREENS.map((id) => $(id)).find((e) => !e.hidden);
  return s ? [...s.querySelectorAll<HTMLElement>("button, select, input")].filter((e) => !(e as HTMLButtonElement).disabled) : [];
}

// ---------- canvas / view (game view: like the real camera, BLUE attacks the top net) ----------
const canvas = $<HTMLCanvasElement>("rink");
const ctx = canvas.getContext("2d")!;
let view: View = fitView(1, 1, P, true);
function resize(): void {
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width));
  canvas.height = Math.max(1, Math.round(r.height));
  view = fitView(canvas.width, canvas.height, P, true);
}
new ResizeObserver(resize).observe(canvas);
resize();

// ---------- input ----------
const held = new Set<string>();
let mouse = { x: 0, y: 0 };
canvas.addEventListener("mousemove", (e) => {
  const r = canvas.getBoundingClientRect();
  mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
});
window.addEventListener("keydown", (e) => {
  if (screen === "match") {
    if (e.code === "Escape") return void toMenu();
    if (state.match_state?.phase === "finished" && (e.code === BINDINGS.rematch || e.code === "NumpadEnter")) {
      e.preventDefault();
      return void startMatch(current!.id, currentSkill);
    }
    const act = actionFor(e.code);
    if (act) {
      e.preventDefault();
      held.add(act);
    }
    return;
  }
  if (e.code === "Escape" && screen === "controls") show("menu");
  if (e.code === "ArrowDown" || e.code === "ArrowUp") {
    const f = focusables();
    const t = e.target as HTMLElement;
    if (t.tagName === "SELECT" || (t as HTMLInputElement).type === "range") return; // arrows change the value there
    const i = f.indexOf(document.activeElement as HTMLElement);
    f[(i + (e.code === "ArrowDown" ? 1 : f.length - 1)) % f.length]?.focus();
    e.preventDefault();
  }
});
window.addEventListener("keyup", (e) => {
  const act = actionFor(e.code);
  if (act) held.delete(act);
});
window.addEventListener("blur", () => held.clear());

function humanInput(): PlayerInput {
  const pl = state.players[0];
  const [wx, wy] = toWorld(view, mouse.x, mouse.y);
  const sr = (held.has("right") ? 1 : 0) - (held.has("left") ? 1 : 0);
  const su = (held.has("up") ? 1 : 0) - (held.has("down") ? 1 : 0);
  return {
    move_x: view.rot ? su : sr,
    move_y: view.rot ? -sr : su,
    stick_target_angle: Math.atan2(wy - pl.y, wx - pl.x),
    dash: held.has("dash"),
    lift: held.has("lift"),
  };
}

// ---------- match ----------
let current: Profile | null = null;
let currentSkill = 100;
let bot: BotDriver | null = null;
let botMessage: string | null = null;
let matchSeed = 1;
const sessions = new Map<string, Promise<import("onnxruntime-web/wasm").InferenceSession>>();
let speed = Math.max(1, Number(qs.get("speed")) || 1);

async function makeBot(p: Profile, sk: number): Promise<BotDriver> {
  botMessage = null;
  if (p.policy) {
    const url = BOTS_DIR + p.policy;
    try {
      if (!sessions.has(url)) sessions.set(url, loadPolicy(url, Sim.obs_dim(), Sim.obs_version()));
      const s = await sessions.get(url)!;
      sim.set_scripted(1, ""); // not scripted
      return new PolicyDriver(sim, 1, s, p.name, p.style, sk, matchSeed);
    } catch (e) {
      sessions.delete(url);
      botMessage = `Could not load the ${p.name} model (${(e as Error).message}). Playing the scripted ${p.fallback} instead.`;
      console.warn(botMessage);
    }
  }
  return new ScriptedDriver(sim, 1, p.scripted ?? p.fallback, sk, matchSeed);
}

async function startMatch(id: string, sk: number): Promise<void> {
  const p = profiles.find((x) => x.id === id) ?? profiles[0];
  current = p;
  currentSkill = sk;
  show("loading");
  $("loading-text").textContent = `Loading ${p.name}…`;
  matchSeed++;
  bot = await makeBot(p, sk);
  sim.start_match(2);
  state = JSON.parse(sim.get_state());
  held.clear();
  frameStats.reset();
  botTrack.reset();
  loop = makeLoop();
  show("match");
  $("loading").hidden = true;
  banner(botMessage);
  canvas.focus();
}

function toMenu(): void {
  banner(null);
  screen = "menu";
  show("menu");
}

function stepOnce(): void {
  if (!bot) return;
  const inputs = [humanInput(), bot.input()];
  sim.step(JSON.stringify(inputs));
  state = JSON.parse(sim.get_state()); // per step: the human's stick aim and the bot stats see every step
  botTrack.update(state);
}
const makeLoop = () => new FixedStepLoop(stepOnce, () => P.timestep, 5 * speed);
let loop = makeLoop();

function showGameOver(): void {
  const m = state.match_state!;
  const [a, b] = state.score;
  const won = m.winner === 0;
  $("result").textContent = won ? "You win!" : `${current!.name} wins`;
  const how = m.overtime ? "in overtime" : Math.abs(a - b) >= P.mercy_lead ? "by the mercy rule" : "at the end of the clock";
  $("result-detail").textContent = `${TEAM_NAMES[0]} ${a} – ${b} ${TEAM_NAMES[1]}, ${how}. Opponent: ${bot!.label}, skill ${currentSkill}.`;
  $("gameover").hidden = false;
  $("rematch").focus();
}

// ---------- verification stats ----------
const frameStats = {
  frames: [] as number[], // ms between frames
  work: [] as number[], // ms spent stepping + drawing per frame
  reset() {
    this.frames = [];
    this.work = [];
  },
};
const botTrack = {
  travel: 0,
  shots: 0,
  last: null as State | null,
  reset() {
    this.travel = 0;
    this.shots = 0;
    this.last = null;
  },
  update(s: State) {
    const l = this.last;
    if (l && l.players.length === 2 && s.match_state?.phase === "playing" && l.match_state?.phase === "playing") {
      const b = s.players[1], lb = l.players[1];
      this.travel += Math.hypot(b.x - lb.x, b.y - lb.y);
      // A bot shot: the puck sped up by > 4 m/s with the bot closer to it than the human.
      const dv = Math.hypot(s.puck.vx, s.puck.vy) - Math.hypot(l.puck.vx, l.puck.vy);
      const db = Math.hypot(s.puck.x - b.x, s.puck.y - b.y), dh = Math.hypot(s.puck.x - s.players[0].x, s.puck.y - s.players[0].y);
      if (dv > 4 && db < dh && db < P.stick_length + 1) this.shots++;
    }
    this.last = s;
  },
};

// ---------- frame ----------
let last = performance.now();
function frame(now: number): void {
  const dt = (now - last) / 1000;
  last = now;
  if (screen === "match" && bot) {
    const t0 = performance.now();
    const finished = state.match_state?.phase === "finished";
    if (!finished) {
      loop.advance(dt * speed);
      if (state.match_state?.phase === "finished") showGameOver();
    }
    draw(ctx, view, P, state, { targets: [], spawn: null, target: null, ghost: null, ghostIndex: 0, forehand: null });
    drawMatchHud(ctx, P, state);
    frameStats.work.push(performance.now() - t0);
    frameStats.frames.push(dt * 1000);
  } else {
    draw(ctx, view, P, state, { targets: [], spawn: null, target: null, ghost: null, ghostIndex: 0, forehand: null });
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---------- verification hooks (read-only views + controls used by scripts/verify_game.mjs) ----------
const pct = (a: number[], q: number) => (a.length ? [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(q * a.length))] : NaN);
declare global {
  interface Window { game: unknown }
}
window.game = {
  get screen() { return screen; },
  get state() { return state; },
  get summary() {
    const m = state.match_state;
    return { score: state.score, clock: m ? formatClock(m.clock_ticks, P.timestep) : null, phase: m?.phase ?? null, overtime: m?.overtime ?? false, winner: m?.winner ?? null, tick: state.tick };
  },
  get bot() {
    if (!bot) return null;
    const ms = bot.stats.inferenceMs;
    return { label: bot.label, message: botMessage, travel: botTrack.travel, shots: botTrack.shots, decisions: bot.stats.decisions, skipped: bot.stats.skipped, mistakes: bot.stats.mistakes,
      inferenceMs: { n: ms.length, mean: ms.reduce((a, b) => a + b, 0) / Math.max(1, ms.length), p50: pct(ms, 0.5), p95: pct(ms, 0.95), max: Math.max(0, ...ms) } };
  },
  get frames() {
    const f = frameStats.frames.slice(5), w = frameStats.work.slice(5);
    return { n: f.length, frameMs: { p50: pct(f, 0.5), p95: pct(f, 0.95) }, workMs: { mean: w.reduce((a, b) => a + b, 0) / Math.max(1, w.length), p95: pct(w, 0.95), max: Math.max(0, ...w) } };
  },
  profiles: profiles.map((p) => ({ id: p.id, name: p.name, policy: p.policy, scripted: p.scripted })),
  setSpeed(n: number) {
    speed = Math.max(1, n);
    loop = makeLoop();
  },
  /** Test hook: make the next goals end the match quickly (state edit: score). */
  setScore(a: number, b: number) {
    const s = JSON.parse(sim.get_state());
    s.score = [a, b];
    sim.set_state(JSON.stringify(s));
    state = s;
  },
};
show("menu"); // initial focus on Play vs Bot
document.body.dataset.ready = "1";
