// Relative-mouse stick check: bursts of movementX through the REAL pointer lock (Playwright mouse.move
// while the canvas holds the lock), per-step stick angle / target / wrapped difference, and a summary of
// stalls, reversals, the lead and the settle time. Also a slow-movement tracking check.
// Run: node scripts/repro_stick.mjs [label]   (own Vite dev server)
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const label = process.argv[2] ?? "";
const PORT = 5188;
const server = spawn(`npx vite --port ${PORT} --strictPort`, { cwd: web, shell: true, stdio: ["ignore", "pipe", "pipe"], detached: process.platform !== "win32" });
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite did not start")), 30000);
  server.stdout.on("data", (d) => { if (String(d).includes("Local:")) { clearTimeout(t); res(); } });
});
const launch = async () => { try { return await chromium.launch(); } catch (e) { const a = process.env.PW_CHROMIUM || "/opt/pw-browsers/chromium"; if (!existsSync(a)) throw e; return chromium.launch({ executablePath: a }); } };
const wrap = (a) => { a = (a + Math.PI) % (2 * Math.PI); if (a < 0) a += 2 * Math.PI; return a - Math.PI; };
const deg = (r) => (r * 180) / Math.PI;
let failed = false;
const browser = await launch();
try {
  const page = await browser.newPage({ viewport: { width: 1400, height: 760 } });
  page.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  await page.goto(`http://localhost:${PORT}/lab.html?mouse=relative`);
  await page.waitForSelector("body[data-ready='1']");
  const ev = (fn, a) => page.evaluate(fn, a);
  // Per-step recorder inside the page.
  await ev(() => {
    window.__rec = [];
    window.lab.lab.stepListeners.push((l) => {
      const p = l.state.players[0];
      window.__rec.push({ tick: l.state.tick, a: p.stick_angle, w: p.stick_omega, t: l.inputs[0].stick_target_angle, mx: window.__mx ?? 0 });
    });
  });
  await page.mouse.move(500, 380);
  await page.mouse.click(500, 380); // capture
  await page.waitForTimeout(200);
  const locked = (await ev(() => window.lab.aim)).locked;
  console.log(`pointer locked: ${locked}${label ? "   [" + label + "]" : ""}`);
  if (!locked) throw new Error("no pointer lock");
  const sens = (await ev(() => window.lab.aim)).sensitivity;
  let x = 500;

  async function burst(units, dir) {
    await ev(() => { window.__rec = []; window.__mx = 0; });
    // 30 frames of movement, one mouse.move per animation frame.
    for (let f = 0; f < 30; f++) {
      x += dir * units;
      await ev((d) => { window.__mx = (window.__mx ?? 0) + d; }, dir * units);
      await page.mouse.move(x, 380);
      await ev(() => new Promise((r) => requestAnimationFrame(() => r())));
    }
    const stopTick = await ev(() => window.lab.lab.state.tick);
    await page.waitForTimeout(1000);
    const rec = await ev(() => window.__rec);
    // Unwrap the stick angle AND the target so both are continuous past ±pi, then remove the whole turns the
    // target may carry from earlier bursts: lead = target − stick, continuous (it may exceed pi before the fix).
    let un = rec[0].a, prev = rec[0].a, tun = rec[0].t, tprev = rec[0].t;
    for (const r of rec) { un += wrap(r.a - prev); prev = r.a; r.aun = un; tun += wrap(r.t - tprev); tprev = r.t; r.tun = tun; }
    const turns = Math.round((rec[0].tun - rec[0].aun) / (2 * Math.PI)) * 2 * Math.PI;
    for (const r of rec) r.t = r.tun - turns;
    const during = rec.filter((r) => r.tick <= stopTick);
    const after = rec.filter((r) => r.tick > stopTick);
    const want = -dir; // screen right = clockwise; in the vertical view a clockwise screen turn = negative world angle
    const stalls = during.filter((r, i) => i > 6 && Math.abs(r.w) < 2 && Math.abs(wrap(r.t - r.a)) > 0.3).length;
    const revSteps = during.filter((r, i) => i > 6 && Math.sign(r.w) === -want && Math.abs(r.w) > 2);
    // A reversal with the target far away (> 90°) is the short-way flip (the bug); one within a few degrees
    // of the target is the sim's hard-stop overshoot (~1–2°, then settle) when mouse events pause briefly.
    const flips = revSteps.filter((r) => Math.abs(wrap(r.t - r.a)) > Math.PI / 2).length;
    const reversals = flips;
    const settleRev = revSteps.length - flips;
    const settleRevMaxDeg = settleRev ? Math.max(...revSteps.filter((r) => Math.abs(wrap(r.t - r.a)) <= Math.PI / 2).map((r) => deg(Math.abs(wrap(r.t - r.a))))) : 0;
    const leads = rec.map((r) => r.t - r.aun);
    const maxLead = Math.max(...leads.map(Math.abs));
    const maxWrapped = Math.max(...rec.map((r) => Math.abs(wrap(r.t - r.a))));
    const leadAtStop = leads[during.length - 1];
    // settle: first step after the stop from which |wrapped diff| < 0.02 and |omega| < 0.5 for the rest
    let settle = null;
    for (let i = 0; i < after.length; i++) {
      if (after.slice(i).every((r) => Math.abs(wrap(r.t - r.a)) < 0.02 && Math.abs(r.w) < 0.5)) { settle = (after[i].tick - stopTick) / 120; break; }
    }
    const meanW = during.slice(7).reduce((s, r) => s + r.w * want, 0) / Math.max(1, during.length - 7);
    const mouseTurn = units * 30 * sens;
    const stickTurn = Math.abs(rec.at(-1).aun - rec[0].aun);
    return { units, dir, steps: during.length, mouseTurnDeg: deg(mouseTurn), stickTurnDeg: deg(stickTurn), meanW, stalls, reversals, settleRev, settleRevMaxDeg, maxLeadDeg: deg(maxLead), maxWrappedDeg: deg(maxWrapped), leadAtStopDeg: deg(Math.abs(leadAtStop)), settle };
  }

  const rows = [];
  for (const u of [20, 60, 150, 400]) for (const d of [1, -1]) rows.push(await burst(u, d));
  console.log("\n| units/frame | dir | mouse turn ° | stick turn ° | mean ω along mouse (rad/s) | stall steps | reverse steps (flip, target > 90° away) | settle-overshoot steps (max off-target °) | max lead ° | max |wrapped diff| ° | lead at stop ° | settle after stop (s) |");
  console.log("|---|---|---|---|---|---|---|---|---|---|---|---|");
  for (const r of rows) console.log(`| ${r.units} | ${r.dir > 0 ? "right" : "left"} | ${r.mouseTurnDeg.toFixed(0)} | ${r.stickTurnDeg.toFixed(0)} | ${r.meanW.toFixed(1)} | ${r.stalls} | ${r.reversals} | ${r.settleRev}${r.settleRev ? ` (${r.settleRevMaxDeg.toFixed(1)}°)` : ""} | ${r.maxLeadDeg.toFixed(1)} | ${r.maxWrappedDeg.toFixed(1)} | ${r.leadAtStopDeg.toFixed(1)} | ${r.settle === null ? "never (>1 s)" : r.settle.toFixed(3)} |`);
  const bad = rows.filter((r) => r.stalls > 0 || r.reversals > 0 || r.settle === null);
  console.log(`\nstall/reversal/no-settle rows: ${bad.length}`);

  // Slow movement: 100 units over ~1 s in 2-unit steps; the stick should follow the integrated mouse angle 1:1.
  await ev(() => { window.__rec = []; });
  const start = await ev(() => ({ a: window.lab.lab.state.players[0].stick_angle, aim: window.lab.aim.worldAngle }));
  console.log(`\nslow: 100 units over ~1 s (2 units per frame), sensitivity ${sens}`);
  console.log("  frame | mouse units | mouse angle ° (world) | stick angle ° | target ° | stick − mouse °");
  let acc = 0;
  const slow = [];
  for (let f = 1; f <= 50; f++) {
    x += 2;
    acc += 2;
    await page.mouse.move(x, 380);
    await page.waitForTimeout(20);
    const s = await ev(() => ({ a: window.lab.lab.state.players[0].stick_angle, t: window.lab.lab.inputs[0].stick_target_angle }));
    const mouseAngle = start.aim - acc * sens; // right = clockwise = negative world in the vertical view
    slow.push({ f, acc, mouseAngle, ...s });
    if (f % 5 === 0) console.log(`  ${String(f).padStart(5)} | ${String(acc).padStart(11)} | ${deg(mouseAngle).toFixed(2).padStart(21)} | ${deg(s.a).toFixed(2).padStart(13)} | ${deg(s.t).toFixed(2).padStart(8)} | ${deg(wrap(s.a - mouseAngle)).toFixed(2).padStart(15)}`);
  }
  await page.waitForTimeout(300);
  const fin = await ev(() => ({ a: window.lab.lab.state.players[0].stick_angle, t: window.lab.lab.inputs[0].stick_target_angle }));
  const finMouse = start.aim - acc * sens;
  const trackErr = Math.max(...slow.map((s) => Math.abs(wrap(s.t - s.mouseAngle))));
  console.log(`  after 0.3 s still: stick ${deg(fin.a).toFixed(2)}°, target ${deg(fin.t).toFixed(2)}°, mouse ${deg(finMouse).toFixed(2)}°; max |target − mouse| during the move ${deg(trackErr).toFixed(3)}°`);
  const slowOk = trackErr < 0.01 && Math.abs(wrap(fin.a - finMouse)) < deg(0.5) * 0 + 0.01;
  console.log(slowOk ? "SLOW OK: target = integrated mouse angle (no lag, no deadzone) and the stick ends on it" : "SLOW NOT OK");
  console.log(bad.length === 0 && slowOk ? "\nSTICK OK" : "\nSTICK NOT OK");
  if (bad.length || !slowOk) failed = true;
} finally {
  await browser.close();
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], { stdio: "ignore" });
  else { try { process.kill(-server.pid, "SIGTERM"); } catch { server.kill(); } }
}
process.exit(failed ? 1 : 0);
