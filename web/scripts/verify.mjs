// Headless feel-lab verification. Starts the Vite dev server, drives the page with Playwright,
// prints traces and writes screenshots + recorded files to web/verify-out/.
// Run: npm run wasm && npm run verify
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(here, "..");
const out = path.join(webDir, "verify-out");
mkdirSync(out, { recursive: true });
const URL_ = "http://localhost:5173/lab.html";

const server = spawn("npx vite --port 5173 --strictPort", {
  cwd: webDir,
  shell: true,
  stdio: ["ignore", "pipe", "pipe"],
  // POSIX: own process group, so the whole tree (shell + vite) can be killed at the end.
  detached: process.platform !== "win32",
});
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite did not start")), 30000);
  server.stdout.on("data", (d) => {
    if (String(d).includes("Local:")) { clearTimeout(t); res(); }
  });
});

const sha = (s) => createHash("sha256").update(s).digest("hex");
const f = (v, d = 3) => (v >= 0 ? " " : "") + v.toFixed(d);
let failed = false;
const check = (cond, msg) => { console.log(`${cond ? "PASS" : "FAIL"}: ${msg}`); if (!cond) failed = true; };

// Use Playwright's own browser when it is installed; otherwise fall back to a preinstalled Chromium
// (PW_CHROMIUM, or /opt/pw-browsers/chromium in the Linux cloud container).
async function launch() {
  try {
    return await chromium.launch();
  } catch (e) {
    const alt = process.env.PW_CHROMIUM || "/opt/pw-browsers/chromium";
    if (!existsSync(alt)) throw e;
    console.log(`(playwright browser missing; using ${alt})`);
    return chromium.launch({ executablePath: alt });
  }
}

let browser;
try {
  browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 760 } });
  page.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  await page.goto(URL_);
  await page.waitForSelector("body[data-ready='1']");
  const ev = (fn, arg) => page.evaluate(fn, arg);
  const w2c = (x, y) => ev(([x, y]) => window.lab.worldToClient(x, y), [x, y]);
  const mouseTo = async (x, y) => { const [cx, cy] = await w2c(x, y); await page.mouse.move(cx, cy); };
  const player0 = () => ev(() => window.lab.lab.state.players[0]);

  // ---- 1. rink + puck at rest + mid-slide ----
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(out, "1-rink.png") });
  const rest = await ev(() => window.lab.lab.state.puck);
  console.log(`\n[rink/rest] puck at rest: x=${rest.x} y=${rest.y} z=${rest.z} v=(${rest.vx},${rest.vy})`);
  await page.screenshot({ path: path.join(out, "2-puck-rest.png"), clip: { x: 300, y: 200, width: 460, height: 360 } });
  await ev(() => { const s = structuredClone(window.lab.lab.state); s.puck.vx = 14; s.puck.vy = 4; window.lab.lab.setState(s); });
  await page.waitForTimeout(450);
  const slide = await ev(() => window.lab.lab.state.puck);
  console.log(`[mid-slide] puck x=${slide.x.toFixed(2)} y=${slide.y.toFixed(2)} speed=${Math.hypot(slide.vx, slide.vy).toFixed(2)} m/s`);
  check(Math.hypot(slide.vx, slide.vy) > 1, "puck is moving in mid-slide screenshot");
  await page.screenshot({ path: path.join(out, "3-puck-mid-slide.png") });

  // ---- 2. stick chases mouse target ----
  console.log("\n[stick chase] player 0 at", (await player0()).x.toFixed(2), (await player0()).y.toFixed(2));
  const p0 = await player0();
  await mouseTo(p0.x + 4, p0.y); // position A: angle 0
  await page.waitForTimeout(800);
  for (const [label, dx, dy] of [["B (+90deg)", 0, 4], ["C (-150deg)", -4 * Math.cos(Math.PI / 6), -2]]) {
    const pl = await player0();
    const tr = ev(() => window.lab.trace(48));
    await mouseTo(pl.x + dx, pl.y + dy);
    const rows = await tr;
    console.log(`mouse -> ${label}:   tick   target    angle    omega`);
    rows.filter((_, i) => i % 4 === 0 || i === rows.length - 1).forEach((r) => console.log(`             ${String(r.tick).padStart(6)} ${f(r.target)} ${f(r.angle)} ${f(r.omega, 2)}`));
    const firstMoving = rows.find((r) => Math.abs(r.target - rows[rows.length - 1].target) < 1e-9);
    const i0 = rows.indexOf(firstMoving);
    const gap = Math.abs(Math.atan2(Math.sin(rows[i0].target - rows[i0].angle), Math.cos(rows[i0].target - rows[i0].angle)));
    const end = rows[rows.length - 1];
    const endGap = Math.abs(Math.atan2(Math.sin(end.target - end.angle), Math.cos(end.target - end.angle)));
    check(gap > 0.5 && endGap < 0.05, `stick lags (gap ${gap.toFixed(2)} rad on the step the target moved) then converges (gap ${endGap.toFixed(4)})`);
  }

  // ---- 3. friction changes slide distance for the same hit ----
  const hit = async () => {
    const P = { x: -15, y: 0 };
    await mouseTo(P.x + 3 * Math.cos(-1.2), P.y + 3 * Math.sin(-1.2));
    await page.waitForTimeout(150);
    await ev((P) => {
      const s = structuredClone(window.lab.lab.state);
      Object.assign(s.players[0], { x: P.x, y: P.y, vx: 0, vy: 0, stick_angle: -1.2, stick_omega: 0 });
      Object.assign(s.puck, { x: P.x + 1.2, y: P.y + 0.2, z: 0, vx: 0, vy: 0, vz: 0 });
      window.lab.lab.setState(s);
    }, P);
    const tr = ev(() => window.lab.trace(900));
    await mouseTo(P.x + 3 * Math.cos(1.2), P.y + 3 * Math.sin(1.2));
    const rows = await tr;
    const start = { x: -15 + 1.2, y: 0.2 };
    const peak = Math.max(...rows.map((r) => r.speed));
    const end = rows[rows.length - 1];
    let pathLen = 0;
    for (let i = 1; i < rows.length; i++) pathLen += Math.hypot(rows[i].x - rows[i - 1].x, rows[i].y - rows[i - 1].y);
    return { peak, pathLen, end, stopped: end.speed === 0, straight: Math.hypot(end.x - start.x, end.y - start.y) };
  };
  const a = await hit();
  await page.fill("#param-ice_friction_decel", "2.5");
  const fr = await ev(() => window.lab.lab.p.ice_friction_decel);
  const b = await hit();
  console.log(`\n[friction] ice_friction_decel=1.0: launch ${a.peak.toFixed(3)} m/s, slide path ${a.pathLen.toFixed(3)} m, stopped=${a.stopped}`);
  console.log(`[friction] ice_friction_decel=${fr} (set via DOM): launch ${b.peak.toFixed(3)} m/s, slide path ${b.pathLen.toFixed(3)} m, stopped=${b.stopped}`);
  check(fr === 2.5, "panel input changed the running sim's friction");
  check(Math.abs(a.peak - b.peak) < 0.05 * a.peak, "same hit (launch speeds within 5%)");
  check(a.pathLen - b.pathLen > 1, "slide distances differ (higher friction slides shorter)");
  await page.click("#params-reset");

  // ---- 4. F spawn height -> gravity arc ----
  await mouseTo(-3, 6);
  await page.keyboard.down("KeyF");
  await page.waitForTimeout(1500);
  const held = await ev(() => ({ spawn: window.lab.lab.spawn, z: window.lab.lab.state.puck.z }));
  await page.screenshot({ path: path.join(out, "4-spawn-high.png") });
  const arc = ev(() => window.lab.trace(150));
  await page.keyboard.up("KeyF");
  const zs = await arc;
  console.log(`\n[gravity] held F 1.5 s: spawn z=${held.spawn.z.toFixed(3)} m, puck z=${held.z.toFixed(3)}; after release:`);
  console.log("   tick      z(m)");
  zs.filter((_, i) => i % 8 === 0 || i === zs.length - 1).forEach((r) => console.log(`  ${String(r.tick).padStart(6)}  ${r.z.toFixed(4)}`));
  const zPeak = Math.max(...zs.map((r) => r.z));
  // First ground contact: the first step whose vertical velocity is no longer negative after falling, near the ice.
  // (Samples are per step and contact happens on a substep, so z === 0 is only seen by chance.)
  const land = zs.findIndex((r, i) => i > 0 && zs[i - 1].vz < 0 && r.vz >= 0 && r.z < 0.3);
  const tFall = land / 120;
  console.log(`  first ground contact after ${land} steps (${tFall.toFixed(3)} s); free fall from ${zPeak.toFixed(3)} m predicts ${Math.sqrt((2 * zPeak) / 9.81).toFixed(3)} s`);
  check(held.spawn.z > 1 && land > 0, "spawn rose while F held and the puck fell to the ice");
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(out, "5-spawn-landed.png") });

  // ---- 5. record, replay, compare trajectory hashes ----
  await ev(() => {
    const s = structuredClone(window.lab.lab.state);
    Object.assign(s.players[0], { x: -8, y: -2, vx: 0, vy: 0 });
    Object.assign(s.puck, { x: -6.6, y: -2, z: 0, vx: 0, vy: 0, vz: 0 });
    window.lab.lab.setState(s);
  });
  await mouseTo(-8, -6);
  await page.waitForTimeout(200);
  await page.click("#rec");
  await mouseTo(-4, 1);
  await page.keyboard.down("KeyD");
  await page.waitForTimeout(400);
  await page.keyboard.down("Space");
  await mouseTo(-2, 4);
  await page.waitForTimeout(300);
  await page.keyboard.up("Space");
  await page.keyboard.down("ShiftLeft");
  await page.waitForTimeout(100);
  await page.keyboard.up("ShiftLeft");
  await page.keyboard.up("KeyD");
  await mouseTo(-10, -4);
  await page.waitForTimeout(700);
  await page.click("#rec");
  const rec = await ev(() => ({ csv: window.lab.toCsv(window.lab.lab.lastRecording.samples), sc: JSON.stringify(window.lab.lab.lastRecording.scenario, null, 2) + "\n" }));
  writeFileSync(path.join(out, "lab_run.csv"), rec.csv);
  writeFileSync(path.join(out, "lab_run.scenario.json"), rec.sc);
  await ev(() => window.lab.lab.lastReplay = null);
  await page.click("#replay");
  await page.waitForFunction(() => window.lab.lab.mode === "play" && window.lab.lab.lastReplay, null, { timeout: 30000 });
  const rep = await ev(() => window.lab.toCsv(window.lab.lab.lastReplay.samples));
  writeFileSync(path.join(out, "lab_run.replay.csv"), rep);
  const sc = JSON.parse(rec.sc);
  const maxZ = Math.max(...rec.csv.trim().split("\n").slice(1).map((l) => +l.split(",")[3]));
  console.log(`\n[record/replay] ${sc.steps} steps, ${sc.inputs.length} input events, puck max z ${maxZ.toFixed(3)} m`);
  console.log(`  recorded trajectory sha256: ${sha(rec.csv)}`);
  console.log(`  replayed trajectory sha256: ${sha(rep)}`);
  check(sha(rec.csv) === sha(rep), "recorded and replayed trajectories are byte-identical");

  // ---- 6. ghost overlay from exported CSV ----
  await page.setInputFiles("#ghost-file", path.join(out, "lab_run.csv"));
  await page.waitForSelector("#ghost-scrub:not([disabled])");
  await page.$eval("#ghost-scrub", (el) => { el.value = String(Math.floor(+el.max * 0.45)); el.dispatchEvent(new Event("input")); });
  await page.waitForTimeout(200);
  console.log(`\n[ghost] loaded ${sc.steps + 1} samples; scrub label: ${await page.textContent("#ghost-t")}`);
  await page.screenshot({ path: path.join(out, "6-ghost.png") });

  // ---- 7. real measured track (measure/) in game view: nets top/bottom, +x up, +y left ----
  const realCsv = path.resolve(webDir, "..", "measure", "results", "justpuckphysicshittingitatfixedpoint.traj.csv");
  if (existsSync(realCsv)) {
    await page.click("#ghost-clear");
    await page.check("#game-view");
    await page.waitForTimeout(100);
    await page.setInputFiles("#ghost-file", realCsv);
    await page.waitForSelector("#ghost-scrub:not([disabled])");
    await page.waitForTimeout(200);
    const rows = readFileSync(realCsv, "utf8").trim().split("\n").slice(1).map((l) => l.split(",").map(Number));
    const P = await ev(() => window.lab.lab.p);
    const xs = rows.map((r) => r[1]);
    const ys = rows.map((r) => r[2]);
    const inX = rows.filter((r) => Math.abs(r[1]) <= P.rink_length / 2).length;
    const inBox = rows.filter((r) => Math.abs(r[1]) <= P.rink_length / 2 && Math.abs(r[2]) <= P.rink_width / 2).length;
    console.log(`\n[real ghost] ${rows.length} samples from ${path.basename(realCsv)}`);
    console.log(`  x range ${Math.min(...xs).toFixed(2)} .. ${Math.max(...xs).toFixed(2)} m (sim rink half length ${P.rink_length / 2})`);
    console.log(`  y range ${Math.min(...ys).toFixed(2)} .. ${Math.max(...ys).toFixed(2)} m (sim rink half width ${P.rink_width / 2})`);
    console.log(`  inside sim rink length: ${inX}/${rows.length}; inside sim rink box: ${inBox}/${rows.length}`);
    const v = await ev(() => window.lab.view);
    const top = await w2c(P.net_x, 0);
    const bot = await w2c(-P.net_x, 0);
    const left = await w2c(0, 5);
    console.log(`  game view: rot=${v.rot}; net +x at client y=${top[1].toFixed(0)}, net -x at y=${bot[1].toFixed(0)}, +y at client x=${left[0].toFixed(0)} (center x ${(await w2c(0, 0))[0].toFixed(0)})`);
    check(v.rot && top[1] < bot[1] && left[0] < (await w2c(0, 0))[0], "game view puts +x (net 0) at the top and +y to the left");
    check(inX === rows.length, "every measured sample lies within the rink length (scale = params rink_length / board px)");
    await page.screenshot({ path: path.join(out, "7-real-ghost-game-view.png") });
    await page.uncheck("#game-view");
  } else {
    console.log(`\n[real ghost] skipped: ${realCsv} not found (run python -m measure track first)`);
  }
} finally {
  await browser?.close();
  // Kill the whole vite process tree (on Windows the shell wrapper would otherwise orphan it).
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], { stdio: "ignore" });
  else {
    try { process.kill(-server.pid, "SIGTERM"); } catch { server.kill(); }
  }
}
console.log(failed ? "\nVERIFY FAILED" : "\nVERIFY OK");
process.exit(failed ? 1 : 0);
