// w3 checks in the lab through the real UI: the Space brake (from top speed, during a dash, released
// mid-stop), lift on E (Space no longer lifts), the lead readout, and record/replay with the relative mouse
// and the brake. Run: npm run verify-w3 (own Vite dev server). Screenshot: web/verify-out/w3-lab-lead.png.
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.join(web, "verify-out");
mkdirSync(out, { recursive: true });
const PORT = 5189;
const server = spawn(`npx vite --port ${PORT} --strictPort`, { cwd: web, shell: true, stdio: ["ignore", "pipe", "pipe"], detached: process.platform !== "win32" });
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite did not start")), 30000);
  server.stdout.on("data", (d) => { if (String(d).includes("Local:")) { clearTimeout(t); res(); } });
});
let failed = false;
const check = (c, m) => { console.log(`${c ? "PASS" : "FAIL"}: ${m}`); if (!c) failed = true; };
const launch = async () => { try { return await chromium.launch(); } catch (e) { const a = process.env.PW_CHROMIUM || "/opt/pw-browsers/chromium"; if (!existsSync(a)) throw e; return chromium.launch({ executablePath: a }); } };
const browser = await launch();
try {
  const page = await browser.newPage({ viewport: { width: 1400, height: 760 } });
  page.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  const ev = (fn, a) => page.evaluate(fn, a);
  await page.goto(`http://localhost:${PORT}/lab.html?mouse=relative`);
  await page.waitForSelector("body[data-ready='1']");
  const P = await ev(() => ({ accel: window.lab.lab.p.player_accel, max: window.lab.lab.p.player_max_speed, dt: window.lab.lab.p.timestep, dash: window.lab.lab.p.dash_speed }));
  await ev(() => {
    window.__rec = [];
    window.lab.lab.stepListeners.push((l) => {
      const p = l.state.players[0], i = l.inputs[0];
      window.__rec.push({ tick: l.state.tick, x: p.x, y: p.y, vx: p.vx, vy: p.vy, mx: i.move_x, my: i.move_y, lift: i.lift, dash: i.dash, dt: p.dash_timer });
    });
  });
  // Put the player at the bottom, at rest, with room to skate up (+x) the rink.
  const park = () => ev(() => { const s = structuredClone(window.lab.lab.state); Object.assign(s.players[0], { x: -24, y: 0, vx: 0, vy: 0 }); Object.assign(s.players[1], { x: 28, y: 15 }); window.lab.lab.setState(s); window.__rec = []; });
  const rec = () => ev(() => window.__rec);
  const sp = (r) => Math.hypot(r.vx, r.vy);

  // ---- 1. top speed with W, then Space (W still held: Space overrides WASD) ----
  await park();
  await page.keyboard.down("KeyW");
  await page.waitForTimeout(1300);
  let r = await rec();
  const top = r.at(-1);
  console.log(`[brake] W held 1.3 s: speed ${sp(top).toFixed(4)} m/s (cap ${P.max})`);
  check(Math.abs(sp(top) - P.max) < 1e-6, "reached top speed");
  await ev(() => { window.__rec = []; });
  await page.keyboard.down("Space");
  await page.waitForTimeout(1200);
  await page.keyboard.up("Space");
  await page.keyboard.up("KeyW");
  r = await rec();
  const start = r.findIndex((q) => Math.hypot(q.mx, q.my) > 0 && q.mx < 0); // first brake input (opposite to +x)
  const brake = r.slice(Math.max(0, start - 1));
  const t0 = brake[0].tick, p0 = brake[0], dir = [top.vx / sp(top), top.vy / sp(top)];
  console.log("   t (s) | speed (m/s) | along (m/s) | sideways (m/s) | move input");
  let next = 0, stopIdx = -1;
  for (const [k, q] of brake.entries()) {
    const t = (q.tick - t0) / 120;
    if (stopIdx < 0 && sp(q) < 1e-9) stopIdx = k;
    if (t >= next || k === stopIdx) {
      console.log(`  ${t.toFixed(3).padStart(6)} | ${sp(q).toFixed(4).padStart(11)} | ${(q.vx * dir[0] + q.vy * dir[1]).toFixed(4).padStart(11)} | ${(-q.vx * dir[1] + q.vy * dir[0]).toFixed(4).padStart(14)} | (${q.mx.toFixed(3)}, ${q.my.toFixed(3)})`);
      next = t + 0.1;
    }
    if (stopIdx >= 0 && k > stopIdx + 2) break;
  }
  const stopT = stopIdx >= 0 ? (brake[stopIdx].tick - t0) / 120 : NaN;
  const dist = stopIdx >= 0 ? Math.hypot(brake[stopIdx].x - p0.x, brake[stopIdx].y - p0.y) : NaN;
  const along = brake.slice(0, stopIdx + 1).map((q) => q.vx * dir[0] + q.vy * dir[1]);
  const side = Math.max(...brake.slice(0, stopIdx + 1).map((q) => Math.abs(-q.vx * dir[1] + q.vy * dir[0])));
  const after = brake.slice(stopIdx + 1, stopIdx + 60); // the steps after the stopping step
  console.log(`[brake] stopped after ${stopT.toFixed(3)} s and ${dist.toFixed(3)} m (constant decel ${P.accel} m/s² predicts ${(P.max / P.accel).toFixed(3)} s, ${(P.max * P.max / (2 * P.accel)).toFixed(3)} m); min along-velocity ${Math.min(...along).toFixed(6)}; max sideways ${side.toExponential(2)} m/s`);
  check(stopIdx > 0 && Math.abs(stopT - P.max / P.accel) < 0.03, "Space stops the player from top speed in max/accel (sim acceleration reused)");
  check(Math.min(...along) >= -1e-9 && side < 1e-9, "no reversal and no sideways drift");
  check(after.every((q) => sp(q) < 1e-9 && q.mx === 0 && q.my === 0), "at rest, the brake sends no move (below 0.05 m/s nothing)");

  // ---- 2. Space released mid-stop: WASD takes over again ----
  await park();
  await page.keyboard.down("KeyW");
  await page.waitForTimeout(1200);
  await ev(() => { window.__rec = []; });
  await page.keyboard.down("Space");
  await page.waitForTimeout(250);
  const mid = (await rec()).at(-1);
  await page.keyboard.up("Space");
  await page.waitForTimeout(300);
  const back = (await rec()).at(-1);
  await page.keyboard.up("KeyW");
  console.log(`[brake] W held, Space for 0.25 s: speed ${sp(mid).toFixed(3)} (move ${mid.mx.toFixed(2)}, ${mid.my.toFixed(2)}); Space released, W still held, 0.3 s later: speed ${sp(back).toFixed(3)} (move ${back.mx}, ${back.my})`);
  check(sp(mid) < P.max - 3 && sp(back) > sp(mid) + 3 && back.mx === 1 && back.my === 0, "releasing Space mid-stop returns control to WASD (accelerates up again)");

  // ---- 3. Space during a dash ----
  await park();
  await page.keyboard.down("KeyW");
  await page.waitForTimeout(400);
  await ev(() => { window.__rec = []; });
  await page.keyboard.down("ShiftLeft");
  await page.waitForTimeout(60);
  await page.keyboard.down("Space");
  await page.waitForTimeout(900);
  await page.keyboard.up("Space");
  await page.keyboard.up("ShiftLeft");
  await page.keyboard.up("KeyW");
  r = await rec();
  const dashStart = r.findIndex((q) => q.dt > 0);
  const peak = Math.max(...r.map(sp));
  const stopD = r.findIndex((q, i) => i > dashStart && sp(q) < 1e-9);
  console.log(`[brake] dash at ${((r[dashStart]?.tick - r[0].tick) / 120).toFixed(3)} s (dash window open), peak ${peak.toFixed(2)} m/s, Space from +0.06 s: stopped ${stopD > 0 ? ((r[stopD].tick - r[dashStart].tick) / 120).toFixed(3) + " s after the dash" : "not within 0.9 s"}`);
  console.log("   t (s) | speed | dash_timer | move");
  for (const q of r.filter((_, i) => i % 6 === 0).slice(0, 14)) console.log(`  ${((q.tick - r[0].tick) / 120).toFixed(3).padStart(6)} | ${sp(q).toFixed(3).padStart(6)} | ${q.dt.toFixed(3)} | (${q.mx.toFixed(2)}, ${q.my.toFixed(2)})${q.dash ? " dash" : ""}`);
  check(dashStart >= 0 && stopD > dashStart, "Space during a dash: the dash still fires, then the brake stops the player");

  // ---- 4. lift on E, Space no longer lifts ----
  await page.mouse.click(700, 380); // focus the page (captures the mouse: relative mode)
  await page.keyboard.down("KeyE");
  await page.waitForTimeout(100);
  const e = await ev(() => window.lab.lab.inputs[0]);
  await page.keyboard.up("KeyE");
  await page.keyboard.down("Space");
  await page.waitForTimeout(100);
  const s = await ev(() => window.lab.lab.inputs[0]);
  await page.keyboard.up("Space");
  console.log(`[keys] E held: lift ${e.lift}; Space held: lift ${s.lift}`);
  check(e.lift === true && s.lift === false, "lift is on E; Space no longer lifts");

  // ---- 5. lead readout at the limit (fast flick), screenshot ----
  await park();
  // The lead can't exceed 1.2 rad and the stick closes that in ~46 ms, longer than Playwright's round trips
  // allow, so the readout is sampled while the motion runs INSIDE the page: 300 units per frame through
  // window.lab.mouseDelta (the handler the locked mousemove calls), the #lead text read in the same frames.
  let x = 700;
  const seen = await ev(() => new Promise((done) => {
    const texts = [];
    let f = 0;
    const tick = () => {
      window.lab.mouseDelta(300, 0);
      texts.push(document.getElementById("lead").textContent);
      if (++f < 20) requestAnimationFrame(tick);
      else done(texts);
    };
    requestAnimationFrame(tick);
  }));
  // Keep the flick going while taking the screenshot so the badge is visible on it.
  await ev(() => { window.__flick = setInterval(() => window.lab.mouseDelta(300, 0), 8); });
  await page.waitForTimeout(80);
  const leadText = await page.textContent("#lead");
  await page.screenshot({ path: path.join(out, "w3-lab-lead.png") });
  await ev(() => clearInterval(window.__flick));
  const shot = /AT LIMIT/.test(leadText) && seen.some((t) => /AT LIMIT/.test(t));
  const sensText = await page.textContent("#sens-readout");
  console.log(`[readout] during the flick: ${seen.slice(0, 6).map((t) => `"${t}"`).join(", ")} …; after: "${await page.textContent("#lead")}"; "${sensText}"`);
  check(shot && /sensitivity/.test(sensText), `the lab readout shows the lead (at the limit during a fast flick: "${leadText}") and the sensitivity`);

  // ---- 6. record + replay with the relative mouse and the brake ----
  await page.keyboard.press("Escape");
  await park();
  await page.click("#rec");
  await page.mouse.click(700, 380);
  await page.keyboard.down("KeyW");
  for (let f = 0; f < 20; f++) { x += f % 3 ? 120 : -200; await page.mouse.move(x, 380); await page.waitForTimeout(30); }
  await page.keyboard.down("Space");
  await page.waitForTimeout(500);
  await page.keyboard.up("Space");
  await page.keyboard.down("KeyE");
  await page.waitForTimeout(120);
  await page.keyboard.up("KeyE");
  await page.keyboard.up("KeyW");
  await page.waitForTimeout(150);
  await page.keyboard.press("Escape");
  await page.click("#rec");
  const recd = await ev(() => ({ csv: window.lab.toCsv(window.lab.lab.lastRecording.samples), steps: window.lab.lab.lastRecording.scenario.steps,
    events: window.lab.lab.lastRecording.scenario.inputs, }));
  await page.click("#replay");
  await page.waitForFunction(() => window.lab.lab.mode === "play" && window.lab.lab.lastReplay, null, { timeout: 30000 });
  const rep = await ev(() => window.lab.toCsv(window.lab.lab.lastReplay.samples));
  const h = (q) => createHash("sha256").update(q).digest("hex");
  // Brake inputs: a move opposite to the motion (−x here) while W (+x) is held.
  const brakeEv = recd.events.filter((q) => q.input.move_x < 0).length;
  console.log(`[record/replay] ${recd.steps} steps, ${recd.events.length} input events (${brakeEv} brake inputs, ${recd.events.filter((q) => q.input.lift).length} with lift)\n  recorded sha256 ${h(recd.csv)}\n  replayed sha256 ${h(rep)}`);
  check(h(recd.csv) === h(rep) && brakeEv > 0, "record → replay with the relative mouse and the brake is byte-identical");
} finally {
  await browser.close();
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], { stdio: "ignore" });
  else { try { process.kill(-server.pid, "SIGTERM"); } catch { server.kill(); } }
}
console.log(failed ? "\nVERIFY W3 FAILED" : "\nVERIFY W3 OK");
process.exit(failed ? 1 : 0);
