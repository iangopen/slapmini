// The game: menu → match vs a bot (sim match rules, human = BLUE with WASD + mouse) → game over.
// The feel lab lives at lab.html (Practice). Wall-clock reads live only here.

import init, { Sim } from "../../wasm-pkg/sim_wasm.js";
import paramsFile from "../../../params.json";
import { actionFor, BINDINGS, MOUSE_BINDINGS } from "../bindings";
import { FixedStepLoop } from "../loop";
import { draw, drawInputBadges, drawMatchHud, drawnRadii, fitView, followView, formatClock, keysToMove, toScreen, toWorld, TEAM_NAMES, type View } from "../render";
import { DEFAULT_SETTINGS, loadSettings, PointerCapture, saveSettings, SENS_MAX, SENS_MIN, SENS_STEP, StickAim, StickRaise } from "../stick";
import { brakeMove } from "../brake";
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
  ["W A S D", "skate (8 directions, screen-relative: W = up the screen)"],
  ["Mouse", "relative (default): click the rink to capture the mouse; moving it right swings the stick right, left swings it left (up/down does nothing). Cursor aim: the stick points at the cursor"],
  ["Shift", "dash (needs a move key) — UNVERIFIED binding"],
  ["E", "lift: hold it while the stick hits the puck — UNVERIFIED binding (was Space)"],
  ["Space", "brake / stop: overrides WASD while held — UNVERIFIED binding; stopping time is a placeholder"],
  ["Enter", "rematch after game over"],
  ["Esc", "release the mouse; Esc again leaves the match"],
].map(([k, d]) => `<li><b>${k}</b> — ${d}</li>`).join("");

// Mouse settings (Controls screen), shared with the lab through localStorage.
const mouseSettings = loadSettings();
const mouseMode = $<HTMLSelectElement>("mouse-mode");
const sens = $<HTMLInputElement>("sensitivity");
sens.min = String(SENS_MIN);
sens.max = String(SENS_MAX);
sens.step = String(SENS_STEP);
function refreshMouse(): void {
  mouseMode.value = mouseSettings.mode;
  sens.value = String(mouseSettings.sensitivity);
  $("sens-value").textContent = `${mouseSettings.sensitivity.toFixed(4)} rad per mouse unit (${((mouseSettings.sensitivity * 100 * 180) / Math.PI).toFixed(0)}° per 100)`;
}
mouseMode.addEventListener("change", () => {
  mouseSettings.mode = mouseMode.value === "cursor" ? "cursor" : "relative";
  saveSettings(mouseSettings);
  refreshMouse();
});
sens.addEventListener("input", () => {
  mouseSettings.sensitivity = +sens.value;
  saveSettings(mouseSettings);
  refreshMouse();
});
$("sens-reset").addEventListener("click", () => {
  mouseSettings.sensitivity = DEFAULT_SETTINGS.sensitivity;
  saveSettings(mouseSettings);
  refreshMouse();
});
refreshMouse();
$("rematch").addEventListener("click", () => void startMatch(current!.id, currentSkill));
$("to-menu").addEventListener("click", () => toMenu());

// Arrow keys move focus through the focusable items of the visible screen (plus Tab, as usual).
function focusables(): HTMLElement[] {
  const s = SCREENS.map((id) => $(id)).find((e) => !e.hidden);
  return s ? [...s.querySelectorAll<HTMLElement>("button, select, input")].filter((e) => !(e as HTMLButtonElement).disabled) : [];
}

// ---------- canvas / view (game view: like the real camera, BLUE attacks the top net) ----------
// Camera (w4): "follow" (default) = the real game's camera, following the human player at a fixed zoom that shows the
// whole attack zone (render.ts followView); "fit" = the whole rink (the pre-w4 view). Stored in localStorage.
type CameraMode = "follow" | "fit";
const CAMERA_KEY = "slapmini.camera.v1";
let cameraMode: CameraMode = (() => {
  try {
    return localStorage.getItem(CAMERA_KEY) === "fit" ? "fit" : "follow";
  } catch {
    return "follow";
  }
})();
if (qs.get("camera") === "fit" || qs.get("camera") === "follow") cameraMode = qs.get("camera") as CameraMode;
const canvas = $<HTMLCanvasElement>("rink");
const ctx = canvas.getContext("2d")!;
let view: View = fitView(1, 1, P, true);
/** The view for this frame: follows player 0 in "follow" mode. All conversions stay in render.ts. */
function updateView(): void {
  const pl = state.players[0];
  view = cameraMode === "follow" && pl ? followView(canvas.width, canvas.height, P, pl.x, pl.y) : fitView(canvas.width, canvas.height, P, true);
}
function resize(): void {
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width));
  canvas.height = Math.max(1, Math.round(r.height));
  updateView();
}
new ResizeObserver(resize).observe(canvas);
resize();
const cameraSel = $<HTMLSelectElement>("camera-mode");
cameraSel.value = cameraMode;
cameraSel.addEventListener("change", () => {
  cameraMode = cameraSel.value === "fit" ? "fit" : "follow";
  try {
    localStorage.setItem(CAMERA_KEY, cameraMode);
  } catch {
    /* storage blocked: this page only */
  }
  updateView();
});

// ---------- input ----------
const held = new Set<string>();
let mouse = { x: 0, y: 0 };
canvas.addEventListener("mousemove", (e) => {
  const r = canvas.getBoundingClientRect();
  mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
});
// Relative stick aim (stick.ts): pointer lock on the first click in a match; movementX turns the stick.
const aim = new StickAim(mouseSettings);
const wantLock = () => screen === "match" && mouseSettings.mode === "relative";
const capture = new PointerCapture(canvas, wantLock, (dx, dy) => aim.move(dx, dy), () => updateLockHint());
function updateLockHint(): void {
  $("lock-hint").hidden = !(wantLock() && !capture.locked && state.match_state?.phase !== "finished");
}
// Raised stick (w4): left button held, both mouse modes; the capturing click does not raise (stick.ts StickRaise).
const raise = new StickRaise(canvas, capture, () => mouseSettings.mode === "relative",
  () => screen === "match" && state.match_state?.phase !== "finished", MOUSE_BINDINGS.stickUp);
window.addEventListener("keydown", (e) => {
  if (screen === "match") {
    if (e.code === "Escape") {
      // The first Esc only releases the captured mouse (the browser does it too); the next one leaves.
      if (capture.locked) return void capture.release();
      if (capture.releasedRecently()) return;
      return void toMenu();
    }
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
  // Space/Shift released over a focused button would "click" it; in a match no control has a say.
  if (screen === "match" && act) e.preventDefault();
});
window.addEventListener("blur", () => held.clear());

/** The inputs the sim received on the last step (verification). */
let lastInputs: PlayerInput[] = [];
let matchesStarted = 0;

function humanInput(): PlayerInput {
  const pl = state.players[0];
  // Space = brake: overrides WASD while held (brake.ts); dash and lift pass through unchanged.
  const [mx, my] = held.has("brake")
    ? brakeMove(pl.vx, pl.vy, P.player_accel, P.timestep)
    : keysToMove(view, held.has("up"), held.has("down"), held.has("left"), held.has("right"));
  let stick: number;
  // Relative aim, held within MAX_LEAD of the stick's actual angle (stick.ts).
  if (mouseSettings.mode === "relative") stick = aim.worldAngle(view, pl.stick_angle);
  else {
    const [wx, wy] = toWorld(view, mouse.x, mouse.y);
    stick = Math.atan2(wy - pl.y, wx - pl.x);
  }
  return { move_x: mx, move_y: my, stick_target_angle: stick, dash: held.has("dash"), lift: held.has("lift"), stick_up: raise.held };
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
  matchesStarted++;
  show("loading");
  $("loading-text").textContent = `Loading ${p.name}…`;
  matchSeed++;
  bot = await makeBot(p, sk);
  sim.start_match(2);
  state = JSON.parse(sim.get_state());
  held.clear();
  aim.reset(view, 0); // stick forward: toward +x, the net BLUE attacks (top of the screen)
  // The Play / Rematch button keeps focus otherwise: Space or Enter would press it again.
  (document.activeElement as HTMLElement | null)?.blur?.();
  frameStats.reset();
  botTrack.reset();
  loop = makeLoop();
  show("match");
  $("loading").hidden = true;
  banner(botMessage);
  updateLockHint();
}

function toMenu(): void {
  capture.release();
  raise.release();
  $("lock-hint").hidden = true;
  banner(null);
  screen = "menu";
  show("menu");
}

function stepOnce(): void {
  if (!bot) return;
  const inputs = [humanInput(), bot.input()];
  lastInputs = inputs;
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
  capture.release(); // the buttons need the cursor
  raise.release();
  updateLockHint();
  $("gameover").hidden = false;
  $("rematch").focus();
}

/** LIFT / BRAKE badges while their keys are held (render.ts). */
function drawLiftBadge(): void {
  drawInputBadges(ctx, canvas.height, held.has("lift"), held.has("brake"), raise.held);
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
    const tgt = lastInputs[0]?.stick_target_angle;
    updateView();
    draw(ctx, view, P, state, { targets: tgt === undefined ? [] : [tgt], spawn: null, target: null, ghost: null, ghostIndex: 0, forehand: null, puckMinPx: 0 });
    drawMatchHud(ctx, P, state);
    drawLiftBadge();
    frameStats.work.push(performance.now() - t0);
    frameStats.frames.push(dt * 1000);
  } else {
    updateView();
    draw(ctx, view, P, state, { targets: [], spawn: null, target: null, ghost: null, ghostIndex: 0, forehand: null, puckMinPx: 0 });
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
      stickUp: !!state.players[bot.player]?.stick_up, upProbMean: bot instanceof PolicyDriver ? bot.upProbSum / Math.max(1, bot.stats.decisions) : null,
      upDecisions: bot instanceof PolicyDriver ? bot.upDecisions : null,
      inferenceMs: { n: ms.length, mean: ms.reduce((a, b) => a + b, 0) / Math.max(1, ms.length), p50: pct(ms, 0.5), p95: pct(ms, 0.95), max: Math.max(0, ...ms) } };
  },
  /** Test hook: force the bot's stick up (screenshots of a raised bot stick); false = the bot's own choice. */
  forceBotStickUp(on: boolean) { if (bot) bot.forceStickUp = on; },
  get frames() {
    const f = frameStats.frames.slice(5), w = frameStats.work.slice(5);
    return { n: f.length, frameMs: { p50: pct(f, 0.5), p95: pct(f, 0.95) }, workMs: { mean: w.reduce((a, b) => a + b, 0) / Math.max(1, w.length), p95: pct(w, 0.95), max: Math.max(0, ...w) } };
  },
  profiles: profiles.map((p) => ({ id: p.id, name: p.name, policy: p.policy, scripted: p.scripted })),
  get lastInputs() { return lastInputs; },
  get matchesStarted() { return matchesStarted; },
  get view() { return view; },
  params: P,
  get aim() {
    return { mode: mouseSettings.mode, sensitivity: mouseSettings.sensitivity, screenAngle: aim.screenAngle, worldAngle: aim.worldAngle(view), locked: capture.locked, hint: !$("lock-hint").hidden };
  },
  /** Test hook: the same handler pointer-locked mousemove events call (for when real pointer lock is unavailable). */
  mouseDelta(dx: number, dy: number) {
    aim.move(dx, dy);
  },
  setSensitivity(v: number) {
    sens.value = String(v);
    sens.dispatchEvent(new Event("input"));
  },
  /** Page coordinates of a world point (through the view transform). */
  worldToClient(x: number, y: number): [number, number] {
    const r = canvas.getBoundingClientRect();
    const [sx, sy] = toScreen(view, x, y);
    return [r.left + sx, r.top + sy];
  },
  get activeElement() { return document.activeElement ? `${document.activeElement.tagName}#${document.activeElement.id}` : null; },
  setSpeed(n: number) {
    speed = Math.max(1, n);
    loop = makeLoop();
  },
  /** w4 test hook: camera mode, the raised stick, the last inputs and the drawn radii. */
  get w4() {
    return { camera: cameraMode, stickUp: raise.held, lastInputs, radii: drawnRadii(view, P, 0), scale: view.s };
  },
  setCamera(m: "follow" | "fit") { cameraMode = m; cameraSel.value = m; updateView(); },
  /** Test hook: replace the sim state (verification pictures and drills). */
  setState(s: State) {
    sim.set_state(JSON.stringify(s));
    state = JSON.parse(sim.get_state());
  },
  /** Test hook: one synchronous sim step of the real match sim with these inputs (scripts/param_audit.mjs). */
  drillStep(inputs: PlayerInput[]) {
    sim.step(JSON.stringify(inputs));
    state = JSON.parse(sim.get_state());
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
