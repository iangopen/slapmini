// Skating check with the default params: hold W from rest (real keyboard events), print player speed over
// sim time, check the ramp reaches player_max_speed in about max/accel seconds, then release and coast.
// Run: npm run wasm && npm run verify-skate   (screenshot: web/verify-out/skate-top-speed.png)
import { freePort } from "./free_port.mjs";
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(here, "..");
const out = path.join(webDir, "verify-out");
mkdirSync(out, { recursive: true });
const PORT = await freePort();
const server = spawn(`npx vite --port ${PORT} --strictPort`, {
  cwd: webDir, shell: true, stdio: ["ignore", "pipe", "pipe"], detached: process.platform !== "win32",
});
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite did not start")), 30000);
  server.stdout.on("data", (d) => { if (String(d).includes("Local:")) { clearTimeout(t); res(); } });
});
let failed = false;
const check = (c, m) => { console.log(`${c ? "PASS" : "FAIL"}: ${m}`); if (!c) failed = true; };
async function launch() {
  try { return await chromium.launch(); } catch (e) {
    const alt = process.env.PW_CHROMIUM || "/opt/pw-browsers/chromium";
    if (!existsSync(alt)) throw e;
    return chromium.launch({ executablePath: alt });
  }
}
let browser;
try {
  browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 760 } });
  page.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  await page.goto(`http://localhost:${PORT}/lab.html`);
  await page.waitForSelector("body[data-ready='1']");
  const ev = (fn, arg) => page.evaluate(fn, arg);
  const P = await ev(() => window.lab.lab.p);
  console.log(`params: player_accel ${P.player_accel}, player_max_speed ${P.player_max_speed}, player_ice_drag ${P.player_ice_drag}, rink_length ${P.rink_length}, timestep ${P.timestep}`);
  // Player 0 at rest; sample (tick, speed) on every animation frame inside the page.
  await ev(() => {
    const s = structuredClone(window.lab.lab.state);
    Object.assign(s.players[0], { x: -20, y: 0, vx: 0, vy: 0 });
    window.lab.lab.setState(s);
    window.__samples = [];
    const tick = () => { const st = window.lab.lab.state; const p = st.players[0]; window.__samples.push([st.tick, Math.hypot(p.vx, p.vy)]); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  });
  await page.mouse.move(700, 380);
  await page.waitForTimeout(300);
  const t0 = await ev(() => window.lab.lab.state.tick);
  await page.keyboard.down("KeyW");
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(out, "skate-top-speed.png") });
  await page.keyboard.up("KeyW");
  await page.waitForTimeout(1500);
  const rows = (await ev(() => window.__samples)).filter(([k]) => k >= t0).map(([k, v]) => [(k - t0) * P.timestep, v]);
  const start = rows.find(([, v]) => v > 1e-6)[0];
  console.log("   t_from_press(s)  speed(m/s)");
  let next = 0;
  for (const [t, v] of rows) if (t >= next) { console.log(`   ${t.toFixed(3).padStart(8)}  ${v.toFixed(3).padStart(8)}`); next = t + 0.1; }
  const top = rows.find(([, v]) => v >= P.player_max_speed - 1e-6);
  const tTop = top[0] - start;
  const want = P.player_max_speed / P.player_accel;
  console.log(`motion starts ${start.toFixed(3)} s after the key event reaches the sim; top speed ${P.player_max_speed} reached ${tTop.toFixed(3)} s later (max/accel = ${want.toFixed(3)} s)`);
  check(Math.abs(tTop - want) < 0.05, "ramp from rest reaches the top speed in max/accel (~0.65 s)");
  check(Math.abs(tTop - 0.7) < 0.1, "top speed in about 0.7 s");
  const peak = Math.max(...rows.map(([, v]) => v));
  check(peak <= P.player_max_speed + 1e-9, `speed never exceeds the cap (peak ${peak.toFixed(4)})`);
  const last = rows[rows.length - 1];
  check(last[1] < peak, `coasts down after release (speed ${last[1].toFixed(3)} at t = ${last[0].toFixed(2)} s)`);
} finally {
  await browser?.close();
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], { stdio: "ignore" });
  else { try { process.kill(-server.pid, "SIGTERM"); } catch { server.kill(); } }
}
console.log(failed ? "\nVERIFY SKATE FAILED" : "\nVERIFY SKATE OK");
process.exit(failed ? 1 : 0);
