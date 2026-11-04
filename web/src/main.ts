// Wiring: wasm init, input, rAF loop, panel, record/replay UI, ghost. Wall-clock reads live only here.

import init, { Sim } from "../wasm-pkg/sim_wasm.js";
import paramsFile from "../../params.json";
import { actionFor, BINDINGS, KEY_HELP } from "./bindings";
import { Lab, parseCsv, toCsv, type Recording } from "./lab";
import { FixedStepLoop } from "./loop";
import { buildParamPanel, download } from "./panel";
import { draw, fitView, toScreen, toWorld, type View } from "./render";
import type { ParamFile, Sample, Scenario } from "./types";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

await init();
const ORIGINAL = paramsFile as ParamFile;
const lab = new Lab(Sim, ORIGINAL, 2);

// ---------- canvas / view ----------
const canvas = $<HTMLCanvasElement>("rink");
const ctx = canvas.getContext("2d")!;
// Game view: rotated like the real game's camera (nets top and bottom, +x up, +y left).
const gameView = $<HTMLInputElement>("game-view");
gameView.checked = new URLSearchParams(location.search).get("view") === "game";
let view: View = fitView(1, 1, lab.p, gameView.checked);
function resize(): void {
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width));
  canvas.height = Math.max(1, Math.round(r.height));
  view = fitView(canvas.width, canvas.height, lab.p, gameView.checked);
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

function onKey(e: KeyboardEvent, down: boolean): void {
  if ((e.target as HTMLElement)?.tagName === "INPUT") return;
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
}
window.addEventListener("keydown", (e) => onKey(e, true));
window.addEventListener("keyup", (e) => onKey(e, false));
window.addEventListener("blur", () => {
  held.clear();
  lab.spawnRelease();
});

/** Human input for player 0: WASD + angle from player to mouse (world frame). */
function readHumanInput(): void {
  const pl = lab.state.players[0];
  if (!pl) return;
  const [wx, wy] = mouseWorld();
  // WASD is screen-relative: right/up on screen, mapped into world axes for the current view.
  const sr = (held.has("right") ? 1 : 0) - (held.has("left") ? 1 : 0);
  const su = (held.has("up") ? 1 : 0) - (held.has("down") ? 1 : 0);
  lab.inputs[0] = {
    move_x: view.rot ? su : sr,
    move_y: view.rot ? -sr : su,
    stick_target_angle: Math.atan2(wy - pl.y, wx - pl.x),
    dash: held.has("dash"),
    lift: held.has("lift"),
  };
}

// ---------- fixed-step loop ----------
const loop = new FixedStepLoop(
  () => {
    if (lab.mode !== "replay") readHumanInput();
    lab.step(lab.p.timestep);
  },
  () => lab.p.timestep,
  5,
);

// ---------- readout ----------
let units: "mph" | "kph" = "mph";
const readout = $("readout");
readout.innerHTML = `
  <div class="row"><span class="big" id="speed">0</span>
    <button id="units" title="switch units">mph</button></div>
  <div>height <span id="height">0.00</span> m &nbsp; tick <span id="tick">0</span></div>
  <div>stick <span id="stick">–</span></div>
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
  if (pl) $("stick").textContent = `angle ${pl.stick_angle.toFixed(2)} → target ${lab.inputs[0].stick_target_angle.toFixed(2)} rad`;
  $("target-info").textContent = lab.target ? `target: closest approach ${Number.isFinite(lab.targetBest) ? lab.targetBest.toFixed(2) + " m" : "–"}` : "";
  const pr = lab.progress;
  $("mode").textContent = lab.mode + (lab.mode !== "play" ? ` ${pr.steps}${pr.total ? "/" + pr.total : ""}` : "");
  $("dropped").textContent = String(loop.dropped);
}

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
  loop.advance(dt);
  draw(ctx, view, lab.p, lab.state, {
    targets: lab.mode === "replay" ? [] : [lab.inputs[0]?.stick_target_angle],
    spawn: lab.spawnHeld || lab.spawn.z > 0 || lab.spawn.x || lab.spawn.y ? lab.spawn : null,
    target: lab.target,
    ghost,
    ghostIndex: +scrub.value,
    forehand: lab.toolsEnabled ? lab.forehandPoint() : null,
  });
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
  BINDINGS,
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
        out.push({ tick: l.state.tick, target: l.inputs[0].stick_target_angle, angle: pl.stick_angle, omega: pl.stick_omega, x: pk.x, y: pk.y, z: pk.z, speed: Math.hypot(pk.vx, pk.vy) });
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
