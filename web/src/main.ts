// Wiring: wasm init, input, rAF loop, panel, record/replay UI, ghost. Wall-clock reads live only here.

import init, { Sim } from "../wasm-pkg/sim_wasm.js";
import paramsFile from "../../params.json";
import { actionFor, BINDINGS, KEY_HELP } from "./bindings";
import { Lab, parseCsv, toCsv, type Recording } from "./lab";
import { FixedStepLoop } from "./loop";
import { buildParamPanel, download } from "./panel";
import { draw, drawInputBadges, drawMatchHud, fitView, formatClock, keysToMove, toScreen, toWorld, TEAM_NAMES, type View } from "./render";
import { loadSettings, MAX_LEAD, PointerCapture, saveSettings, SENS_MAX, SENS_MIN, SENS_STEP, StickAim } from "./stick";
import { brakeMove } from "./brake";
import type { ParamFile, Sample, Scenario } from "./types";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const qs = new URLSearchParams(location.search);

await init();
const ORIGINAL = paramsFile as ParamFile;
const lab = new Lab(Sim, ORIGINAL, 2);

// ---------- mouse model (stick.ts; settings shared with the game through localStorage) ----------
const mouseSettings = loadSettings();
// `?mouse=cursor` / `?mouse=relative` override the stored mode for this page load (the verify scripts use cursor aim).
const mouseParam = qs.get("mouse");
if (mouseParam === "cursor" || mouseParam === "relative") mouseSettings.mode = mouseParam;
const aim = new StickAim(mouseSettings);

// ---------- canvas / view ----------
// "Rink vertical" (default on, like the real game's camera and the game page: +x up, BLUE attacks the top
// net, +y left). Off = the old horizontal view (`?view=horizontal`), kept for old recordings and scripts.
// The transform itself lives in render.ts (toScreen / toWorld and the direction helpers).
const canvas = $<HTMLCanvasElement>("rink");
const ctx = canvas.getContext("2d")!;
const gameView = $<HTMLInputElement>("game-view");
gameView.checked = qs.get("view") !== "horizontal" && qs.get("view") !== "lab";
let view: View = fitView(1, 1, lab.p, gameView.checked);
aim.reset(view, 0); // forward = +x
function resize(): void {
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width));
  canvas.height = Math.max(1, Math.round(r.height));
  const world = aim.worldAngle(view);
  view = fitView(canvas.width, canvas.height, lab.p, gameView.checked);
  aim.reset(view, world); // the stick keeps pointing the same way in the world when the view turns
}
gameView.addEventListener("change", resize);
new ResizeObserver(resize).observe(canvas);
resize();

// ---------- input ----------
const held = new Set<string>();
let mouse = { x: canvas.width / 2, y: canvas.height / 2 };
const mouseWorld = () => toWorld(view, mouse.x, mouse.y);
canvas.addEventListener("mousemove", (e) => {
  const r = canvas.getBoundingClientRect();
  mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
});
// Relative aim: the first click on the rink captures the pointer (not during a replay). While captured,
// the practice tools (F, G) use the last cursor position from before the capture.
const wantLock = () => mouseSettings.mode === "relative" && lab.mode !== "replay";
const capture = new PointerCapture(canvas, wantLock, (dx, dy) => aim.move(dx, dy), () => updateLockHint());
function updateLockHint(): void {
  $("lock-hint").hidden = !(wantLock() && !capture.locked);
}
canvas.addEventListener("mousedown", () => (document.activeElement as HTMLElement | null)?.blur?.());
const mouseMode = $<HTMLSelectElement>("mouse-mode");
const sens = $<HTMLInputElement>("sensitivity");
sens.min = String(SENS_MIN);
sens.max = String(SENS_MAX);
sens.step = String(SENS_STEP);
function refreshMouse(): void {
  mouseMode.value = mouseSettings.mode;
  sens.value = String(mouseSettings.sensitivity);
  $("sens-value").textContent = `${mouseSettings.sensitivity.toFixed(4)} rad/unit`;
  updateLockHint();
}
mouseMode.addEventListener("change", () => {
  mouseSettings.mode = mouseMode.value === "cursor" ? "cursor" : "relative";
  if (mouseSettings.mode === "cursor") capture.release();
  saveSettings(mouseSettings);
  refreshMouse();
});
sens.addEventListener("input", () => {
  mouseSettings.sensitivity = +sens.value;
  saveSettings(mouseSettings);
  refreshMouse();
});
refreshMouse();

/** Text-entry fields keep their keys; checkboxes, sliders, selects and buttons do not get game keys.
 * (Before w2 any focused <input> swallowed every key: after clicking a panel checkbox, Space toggled
 * the checkbox instead of lifting.) */
function typing(t: EventTarget | null): boolean {
  const el = t as HTMLElement | null;
  if (!el) return false;
  if (el.tagName === "TEXTAREA" || el.isContentEditable) return true;
  return el.tagName === "INPUT" && ["text", "number", "search", "email", "password", "url"].includes((el as HTMLInputElement).type);
}
function onKey(e: KeyboardEvent, down: boolean): void {
  if (typing(e.target)) return;
  if (down && e.code === "Escape" && capture.locked) return void capture.release(); // the browser does this too
  const act = actionFor(e.code);
  if (!act) return;
  e.preventDefault();
  if (down && e.repeat) return;
  if (down) held.add(act);
  else held.delete(act);
  const [wx, wy] = mouseWorld();
  if (act === "spawn") down ? lab.spawnPress(wx, wy) : lab.spawnRelease();
  if (!down) return;
  if (act === "target") lab.setTarget(wx, wy);
  if (act === "reset") lab.resetPuck();
  if (act === "call") lab.callPuck();
  if (act === "rematch") lab.rematch();
}
window.addEventListener("keydown", (e) => onKey(e, true));
window.addEventListener("keyup", (e) => onKey(e, false));
window.addEventListener("blur", () => {
  held.clear();
  lab.spawnRelease();
});

/** Human input for player 0: WASD (screen-relative, through the view transform) + the stick aim. */
function readHumanInput(): void {
  const pl = lab.state.players[0];
  if (!pl) return;
  // Space = brake: overrides WASD while held (brake.ts); dash and lift pass through unchanged.
  const [mx, my] = held.has("brake")
    ? brakeMove(pl.vx, pl.vy, lab.p.player_accel, lab.p.timestep)
    : keysToMove(view, held.has("up"), held.has("down"), held.has("left"), held.has("right"));
  let stick: number;
  // Relative aim, held within MAX_LEAD of the stick's actual angle (stick.ts).
  if (mouseSettings.mode === "relative") stick = aim.worldAngle(view, pl.stick_angle);
  else {
    const [wx, wy] = mouseWorld();
    stick = Math.atan2(wy - pl.y, wx - pl.x);
  }
  lab.inputs[0] = { move_x: mx, move_y: my, stick_target_angle: stick, dash: held.has("dash"), lift: held.has("lift") };
}

// ---------- fixed-step loop ----------
// `?speed=N` (or window.lab.setSpeed) runs the sim N times faster than real time (verification of whole matches).
let speed = Math.max(1, Number(new URLSearchParams(location.search).get("speed")) || 1);
const makeLoop = () =>
  new FixedStepLoop(
    () => {
      if (lab.mode !== "replay") readHumanInput();
      lab.step(lab.p.timestep);
    },
    () => lab.p.timestep,
    5 * speed,
  );
let loop = makeLoop();

// ---------- readout ----------
let units: "mph" | "kph" = "mph";
const readout = $("readout");
readout.innerHTML = `
  <div class="row"><span class="big" id="speed">0</span>
    <button id="units" title="switch units">mph</button></div>
  <div>height <span id="height">0.00</span> m &nbsp; tick <span id="tick">0</span></div>
  <div>stick <span id="stick">–</span></div>
  <div>lead <span id="lead">–</span> &nbsp; <span id="sens-readout" class="muted"></span></div>
  <div id="target-info" class="muted"></div>
  <div class="muted">mode <span id="mode">play</span> &nbsp; dropped <span id="dropped">0</span></div>`;
$("units").addEventListener("click", () => {
  units = units === "mph" ? "kph" : "mph";
  $("units").textContent = units;
});
$("keys").innerHTML = KEY_HELP.map(([k, d]) => `<li><b>${k}</b> ${d}</li>`).join("");

function updateReadout(): void {
  const pk = lab.state.puck;
  const ms = Math.hypot(pk.vx, pk.vy, pk.vz) * lab.p.speed_display_scale;
  $("speed").textContent = `${(units === "mph" ? ms * 2.236936 : ms * 3.6).toFixed(1)} ${units}`;
  $("height").textContent = pk.z.toFixed(2);
  $("tick").textContent = String(lab.state.tick);
  const pl = lab.state.players[0];
  if (pl) {
    $("stick").textContent = `angle ${pl.stick_angle.toFixed(2)} → target ${lab.inputs[0].stick_target_angle.toFixed(2)} rad`;
    // Lead = target − stick (wrapped), the amount the mouse is ahead of the stick. At the limit you are outrunning it.
    let d = (lab.inputs[0].stick_target_angle - pl.stick_angle + Math.PI) % (2 * Math.PI);
    if (d < 0) d += 2 * Math.PI;
    d -= Math.PI;
    // AT LIMIT = the aim was pulled back to MAX_LEAD on the last step (the lead shown is after the step,
    // when the stick has already closed part of it).
    const atLimit = mouseSettings.mode === "relative" && aim.clamped;
    $("lead").textContent = `${((d * 180) / Math.PI).toFixed(1)}°${mouseSettings.mode === "relative" ? ` (limit ±${((MAX_LEAD * 180) / Math.PI).toFixed(0)}°${atLimit ? ", AT LIMIT" : ""})` : ""}`;
    $("lead").style.color = atLimit ? "var(--warn)" : "";
  }
  $("sens-readout").textContent = `${mouseSettings.mode} · sensitivity ${mouseSettings.sensitivity.toFixed(4)} rad/unit`;
  $("target-info").textContent = lab.target ? `target: closest approach ${Number.isFinite(lab.targetBest) ? lab.targetBest.toFixed(2) + " m" : "–"}` : "";
  const pr = lab.progress;
  $("mode").textContent = lab.mode + (lab.mode !== "play" ? ` ${pr.steps}${pr.total ? "/" + pr.total : ""}` : "");
  $("dropped").textContent = String(loop.dropped);
  updateMatchInfo();
}

// ---------- match mode ----------
const matchBox = $<HTMLInputElement>("match-mode");
const matchRestart = $<HTMLButtonElement>("match-restart");
const PHASE_LABEL: Record<string, string> = { countdown: "countdown", playing: "playing", goal_pause: "goal pause", finished: "finished" };
matchBox.addEventListener("change", () => {
  const ok = matchBox.checked ? lab.startMatch() : lab.endMatch();
  if (!ok) matchBox.checked = lab.inMatch; // not while recording/replaying
});
matchRestart.addEventListener("click", () => lab.startMatch());
function updateMatchInfo(): void {
  const m = lab.state.match_state;
  matchBox.checked = !!m;
  matchBox.disabled = lab.mode !== "play";
  matchRestart.disabled = !m || lab.mode !== "play";
  $("match-info").textContent = m
    ? `${TEAM_NAMES[0]} ${lab.state.score[0]} – ${lab.state.score[1]} ${TEAM_NAMES[1]} · ${formatClock(m.clock_ticks, lab.p.timestep)} · ${PHASE_LABEL[m.phase]}` +
      (m.overtime ? " · OVERTIME" : "") + (m.winner !== null ? ` · ${TEAM_NAMES[m.winner]} wins (Enter = rematch)` : "")
    : "free play (no rules)";
}
if (new URLSearchParams(location.search).get("mode") === "match") lab.startMatch();

// ---------- record / replay ----------
const recBtn = $<HTMLButtonElement>("rec");
const replayBtn = $<HTMLButtonElement>("replay");
const recStatus = $("rec-status");
let loaded: Scenario | null = null;
function recordingReady(r: Recording | null): void {
  const ok = !!r;
  replayBtn.disabled = !ok && !loaded;
  $<HTMLButtonElement>("dl-scenario").disabled = !ok;
  $<HTMLButtonElement>("dl-csv").disabled = !ok;
}
recBtn.addEventListener("click", () => {
  if (lab.mode === "record") {
    const r = lab.stopRecording();
    recBtn.textContent = "Record";
    recBtn.classList.remove("on");
    recStatus.textContent = r ? `recorded ${r.scenario.steps} steps, ${r.scenario.inputs.length} input events` : "";
    loaded = null;
    recordingReady(r);
  } else if (lab.mode === "play") {
    lab.startRecording();
    recBtn.textContent = "Stop";
    recBtn.classList.add("on");
    recStatus.textContent = "recording… (practice tools disabled)";
  }
});
replayBtn.addEventListener("click", () => {
  const sc = loaded ?? lab.lastRecording?.scenario;
  if (!sc) return;
  lab.startReplay(sc);
  recStatus.textContent = `replaying ${sc.name}…`;
});
lab.onReplayDone = () => {
  recStatus.textContent = `replay done: ${lab.lastReplay!.samples.length} samples`;
};
$("dl-scenario").addEventListener("click", () => {
  const r = lab.lastRecording;
  if (r) download(`${r.scenario.name}.scenario.json`, JSON.stringify(r.scenario, null, 2) + "\n");
});
$("dl-csv").addEventListener("click", () => {
  const r = lab.lastRecording;
  if (r) download(`${r.scenario.name}.csv`, toCsv(r.samples), "text/csv");
});
$<HTMLInputElement>("load-scenario").addEventListener("change", async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  loaded = JSON.parse(await f.text()) as Scenario;
  recStatus.textContent = `loaded ${loaded.name} (${loaded.steps} steps)`;
  replayBtn.disabled = false;
});

// ---------- ghost ----------
let ghost: Sample[] | null = null;
const scrub = $<HTMLInputElement>("ghost-scrub");
$<HTMLInputElement>("ghost-file").addEventListener("change", async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  ghost = parseCsv(await f.text());
  scrub.max = String(Math.max(0, ghost.length - 1));
  scrub.value = scrub.max;
  scrub.disabled = false;
  $<HTMLButtonElement>("ghost-clear").disabled = false;
  scrub.dispatchEvent(new Event("input"));
});
scrub.addEventListener("input", () => {
  const g = ghost?.[+scrub.value];
  $("ghost-t").textContent = g ? `t=${g.t.toFixed(2)}s z=${g.z.toFixed(2)}` : "–";
});
$("ghost-clear").addEventListener("click", () => {
  ghost = null;
  scrub.disabled = true;
  $<HTMLButtonElement>("ghost-clear").disabled = true;
  $("ghost-t").textContent = "–";
});

// ---------- params panel ----------
const refreshPanel = buildParamPanel($("params"), ORIGINAL, () => lab.file, (n, v) => {
  lab.setParam(n, v);
  if (n.startsWith("rink_")) resize();
});
$("params-reset").addEventListener("click", () => {
  lab.setParamFile(ORIGINAL);
  refreshPanel();
  resize();
});
$("params-export").addEventListener("click", () => download("params.json", JSON.stringify(lab.file, null, 2) + "\n"));

// ---------- frame ----------
let last = performance.now();
function frame(now: number): void {
  const dt = (now - last) / 1000;
  last = now;
  loop.advance(dt * speed);
  draw(ctx, view, lab.p, lab.state, {
    targets: lab.mode === "replay" ? [] : [lab.inputs[0]?.stick_target_angle],
    spawn: !lab.inMatch && (lab.spawnHeld || lab.spawn.z > 0 || lab.spawn.x || lab.spawn.y) ? lab.spawn : null,
    target: lab.inMatch ? null : lab.target,
    ghost,
    ghostIndex: +scrub.value,
    forehand: lab.toolsEnabled ? lab.forehandPoint() : null,
  });
  drawMatchHud(ctx, lab.p, lab.state);
  drawInputBadges(ctx, canvas.height, held.has("lift"), held.has("brake"));
  updateReadout();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---------- verification hooks (read-only views + state setup; used by scripts/verify.mjs) ----------
declare global {
  interface Window { lab: unknown }
}
window.lab = {
  lab,
  /** Run the sim `n` times faster than real time (the per-frame step cap scales with it). */
  setSpeed(n: number) {
    speed = Math.max(1, n);
    loop = makeLoop();
  },
  formatClock,
  BINDINGS,
  get aim() {
    return { mode: mouseSettings.mode, sensitivity: mouseSettings.sensitivity, screenAngle: aim.screenAngle, worldAngle: aim.worldAngle(view), locked: capture.locked, hint: !$("lock-hint").hidden };
  },
  /** Test hook: the same handler pointer-locked mousemove events call. */
  mouseDelta(dx: number, dy: number) {
    aim.move(dx, dy);
  },
  setMouseMode(m: "relative" | "cursor") {
    mouseMode.value = m;
    mouseMode.dispatchEvent(new Event("change"));
  },
  get view() {
    return view;
  },
  toCsv,
  /** Canvas-relative world → page client coordinates. */
  worldToClient(x: number, y: number): [number, number] {
    const r = canvas.getBoundingClientRect();
    const [sx, sy] = toScreen(view, x, y);
    return [r.left + sx, r.top + sy];
  },
  /** Collect per-step samples for the next n steps. */
  trace(n: number): Promise<object[]> {
    return new Promise((resolve) => {
      const out: object[] = [];
      const f = (l: Lab) => {
        const pl = l.state.players[0];
        const pk = l.state.puck;
        out.push({ tick: l.state.tick, target: l.inputs[0].stick_target_angle, angle: pl.stick_angle, omega: pl.stick_omega, x: pk.x, y: pk.y, z: pk.z, vz: pk.vz, speed: Math.hypot(pk.vx, pk.vy) });
        if (out.length >= n) {
          l.stepListeners = l.stepListeners.filter((g) => g !== f);
          resolve(out);
        }
      };
      lab.stepListeners.push(f);
    });
  },
};
document.body.dataset.ready = "1";
