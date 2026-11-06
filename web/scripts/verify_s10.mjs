// s10 feel check with the default params, real keyboard and mouse events in the feel lab:
//  1. skate from rest and release: speed over time, hard cap, speed-proportional coast;
//  2. Shift alone (nothing happens), dash from rest, dash at top speed (set-speed dash, not an impulse);
//  3. a stick flick (mouse jumps to the other side): speed-up vs the much harder stop.
// Prints the curves and writes web/verify-out/s10-lab.png (lab mid-dash) and s10-curves.png (the curves).
// Run: npm run wasm && npm run verify-s10
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
const table = (rows, every, cols) => {
  let next = -1;
  for (const r of rows) if (r.t >= next) { console.log("   " + cols.map((c) => r[c].toFixed(3).padStart(9)).join(" ")); next = r.t + every; }
};

let browser;
try {
  browser = await launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 760 } });
  page.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  await page.goto(`http://localhost:${PORT}/lab.html?mouse=cursor`); // aims the stick with the cursor
  await page.waitForSelector("body[data-ready='1']");
  const ev = (fn, arg) => page.evaluate(fn, arg);
  const P = await ev(() => window.lab.lab.p);
  console.log(`params: player_accel ${P.player_accel}, player_max_speed ${P.player_max_speed}, player_coast_rate ${P.player_coast_rate}, ` +
    `player_ice_drag ${P.player_ice_drag}, dash_speed ${P.dash_speed}, dash_ramp_time ${P.dash_ramp_time}, ` +
    `stick_max_ang_speed ${P.stick_max_ang_speed}, accel ${P.stick_max_ang_accel}, decel ${P.stick_max_ang_decel}`);
  const rot = await ev(() => window.lab.view.rot);
  const plusX = rot ? "KeyW" : "KeyD"; // the key that moves the player toward +x in this view
  // Per-step samples of player 0 (speed, stick) from the lab's step listeners.
  await ev(() => {
    window.__s = [];
    window.lab.lab.stepListeners.push((l) => {
      const p = l.state.players[0];
      window.__s.push({ tick: l.state.tick, v: Math.hypot(p.vx, p.vy), a: p.stick_angle, w: p.stick_omega, dt: p.dash_timer });
    });
  });
  const place = (x) => ev((x) => {
    const s = structuredClone(window.lab.lab.state);
    Object.assign(s.players[0], { x, y: 0, vx: 0, vy: 0, stick_omega: 0, dash_timer: 0, dash_cooldown: 0 });
    s.puck.x = 0; s.puck.y = 0.42 * window.lab.lab.p.rink_width; s.puck.vx = 0; s.puck.vy = 0;
    window.lab.lab.setState(s);
  }, x);
  const since = async (t0) => (await ev(() => window.__s)).filter((r) => r.tick >= t0).map((r) => ({ ...r, t: (r.tick - t0) * P.timestep }));
  const tick = () => ev(() => window.lab.lab.state.tick);
  const plr = () => ev(() => window.lab.lab.state.players[0]);
  // Mouse straight "ahead" (+x of the player) so the stick stays still while skating.
  const aim = async (dx, dy) => { const p = await plr(); const [cx, cy] = await ev(([x, y]) => window.lab.worldToClient(x, y), [p.x + dx, p.y + dy]); await page.mouse.move(cx, cy); };
  const curves = {};

  // ---- 1. skate and release
  await place(-30);
  await aim(4, 0);
  await page.waitForTimeout(300);
  let t0 = await tick();
  await page.keyboard.down(plusX);
  await page.waitForTimeout(1200);
  await page.keyboard.up(plusX);
  const tRel = (await tick() - t0) * P.timestep;
  await page.waitForTimeout(1600);
  let rows = await since(t0);
  console.log(`\n1. skate from rest (${plusX}) for 1.2 s, release at t = ${tRel.toFixed(3)} s:\n      t (s)  speed (m/s)`);
  table(rows, 0.1, ["t", "v"]);
  curves.skate = rows;
  const start = rows.find((r) => r.v > 1e-6).t;
  const top = rows.find((r) => r.v >= P.player_max_speed - 1e-6);
  check(top && Math.abs(top.t - start - P.player_max_speed / P.player_accel) < 0.03, `top speed after ${(top.t - start).toFixed(3)} s (max/accel ${(P.player_max_speed / P.player_accel).toFixed(3)} s)`);
  check(Math.max(...rows.map((r) => r.v)) <= P.player_max_speed + 1e-9, "never above the cap while skating");
  const relIdx = rows.findIndex((r, i) => i > 0 && r.v < rows[i - 1].v - 1e-9);
  const tr = rows[relIdx - 1].t;
  const at = (t) => rows.reduce((b, r) => (Math.abs(r.t - t) < Math.abs(b.t - t) ? r : b)).v;
  const [c0, c1, c2] = [at(tr), at(tr + 0.6), at(tr + 1.2)];
  console.log(`   coast: ${c0.toFixed(3)} -> ${c1.toFixed(3)} -> ${c2.toFixed(3)} m/s over two 0.6 s intervals: ratios ${(c1 / c0).toFixed(4)} / ${(c2 / c1).toFixed(4)}, drops ${(c0 - c1).toFixed(3)} / ${(c1 - c2).toFixed(3)} m/s`);
  check(Math.abs(c1 / c0 - c2 / c1) < 0.01, "coast decays in proportion to speed (equal ratios over equal times)");
  check(Math.abs(c1 / c0 - Math.exp(-P.player_coast_rate * 0.6)) < 0.01, `ratio per 0.6 s = exp(-player_coast_rate * 0.6) = ${Math.exp(-P.player_coast_rate * 0.6).toFixed(4)}`);

  // ---- 2a. Shift alone
  await place(-30);
  await page.waitForTimeout(200);
  t0 = await tick();
  await page.keyboard.down("ShiftLeft");
  await page.waitForTimeout(400);
  await page.keyboard.up("ShiftLeft");
  rows = await since(t0);
  check(rows.every((r) => r.v === 0 && r.dt === 0), "Shift alone at rest does nothing (no motion, no dash window)");

  // ---- 2b. dash from rest: press the move key and Shift together
  await place(-30);
  await page.waitForTimeout(200);
  t0 = await tick();
  await page.keyboard.down(plusX);
  await page.keyboard.down("ShiftLeft");
  await page.waitForTimeout(90);
  await page.screenshot({ path: path.join(out, "s10-lab.png") });
  await page.waitForTimeout(400);
  await page.keyboard.up("ShiftLeft");
  await page.keyboard.up(plusX);
  await page.waitForTimeout(300);
  rows = await since(t0);
  const go = rows.find((r) => r.v > 1e-6).t;
  rows = rows.map((r) => ({ ...r, t: r.t - go }));
  console.log("\n2b. dash from rest (move key + Shift):\n      t (s)  speed (m/s)");
  table(rows.filter((r) => r.t >= -0.02 && r.t < 0.6), 0.025, ["t", "v"]);
  curves.dashRest = rows;
  const pk = rows.reduce((b, r) => (r.v > b.v ? r : b));
  check(Math.abs(pk.v - P.dash_speed) < 1e-6 && Math.abs(pk.t - P.dash_ramp_time) < 0.03, `peaks at dash_speed ${pk.v.toFixed(3)} m/s after ${pk.t.toFixed(3)} s (ramp ${P.dash_ramp_time} s)`);
  check(rows.find((r) => r.t > 0 && r.t < 0.01)?.v < 0.3 * P.dash_speed, "a ramp, not an instant impulse");

  // ---- 2c. dash at top speed
  await place(-30);
  await page.waitForTimeout(200);
  t0 = await tick();
  await page.keyboard.down(plusX);
  await page.waitForTimeout(1000);
  const tD = (await tick() - t0) * P.timestep;
  await page.keyboard.down("ShiftLeft");
  await page.waitForTimeout(500);
  await page.keyboard.up("ShiftLeft");
  await page.keyboard.up(plusX);
  rows = (await since(t0)).map((r) => ({ ...r, t: r.t - tD }));
  console.log("\n2c. dash at top speed (Shift pressed at t = 0):\n      t (s)  speed (m/s)");
  table(rows.filter((r) => r.t >= -0.1 && r.t < 0.45), 0.025, ["t", "v"]);
  curves.dashTop = rows;
  const pk2 = Math.max(...rows.map((r) => r.v));
  check(pk2 <= P.dash_speed + 1e-6 && pk2 > P.player_max_speed + 1, `peak ${pk2.toFixed(3)} m/s: +${(pk2 - P.player_max_speed).toFixed(2)} over the cap (old additive model: +${P.dash_speed.toFixed(2)})`);
  check(Math.abs(rows.filter((r) => r.t > 0.35 && r.t < 0.45).at(-1).v - P.player_max_speed) < 1e-6, "back at the cap within 0.35 s");

  // ---- 3. stick flick: mouse from +x of the player to its -x side (half a turn)
  await place(-10);
  await aim(4, 0.3);
  await page.waitForTimeout(500);
  t0 = await tick();
  await aim(-4, -0.3);
  await page.waitForTimeout(500);
  rows = await since(t0);
  const mv = rows.findIndex((r) => Math.abs(r.w) > 1e-9);
  rows = rows.map((r) => ({ ...r, t: r.t - rows[mv].t }));
  for (let i = 1; i < rows.length; i++) rows[i].alpha = (Math.abs(rows[i].w) - Math.abs(rows[i - 1].w)) / P.timestep;
  rows[0].alpha = 0;
  console.log("\n3. stick flick (half a turn):\n      t (s)  angle (rad)  omega (rad/s)  d|omega|/dt");
  table(rows.filter((r) => r.t >= -0.01 && r.t < 0.2), 0, ["t", "a", "w", "alpha"]);
  curves.flick = rows;
  const wpk = Math.max(...rows.map((r) => Math.abs(r.w)));
  const up = Math.max(...rows.map((r) => r.alpha));
  const down = -Math.min(...rows.map((r) => r.alpha));
  const stopT = rows.find((r, i) => i > mv && r.t > 0 && Math.abs(r.w) < 1e-9)?.t;
  console.log(`   peak |omega| ${wpk.toFixed(3)} rad/s; largest speed-up ${up.toFixed(0)} rad/s^2, largest slow-down ${down.toFixed(0)} rad/s^2; at rest after ${stopT?.toFixed(3)} s`);
  check(wpk <= P.stick_max_ang_speed + 1e-9 && wpk > P.stick_max_ang_speed - 1e-6, "reaches the hard angular speed cap");
  check(up <= P.stick_max_ang_accel + 1e-6 && down > 2 * up, "stops much harder than it starts");

  // ---- curves image (SVG drawn in a blank page)
  const svg = await ev((c) => {
    const W = 1200, H = 760, pad = 50;
    const panel = (x0, y0, w, h, series, tmin, tmax, vmax, title, ylab) => {
      const X = (t) => x0 + ((t - tmin) / (tmax - tmin)) * w, Y = (v) => y0 + h - (v / vmax) * h;
      let s = `<rect x="${x0}" y="${y0}" width="${w}" height="${h}" fill="#fff" stroke="#ccc"/>`;
      s += `<text x="${x0}" y="${y0 - 8}" font-size="14" font-family="sans-serif">${title}</text>`;
      s += `<text x="${x0 - 40}" y="${y0 + 12}" font-size="11" font-family="sans-serif" fill="#555">${ylab}</text>`;
      for (let v = 0; v <= vmax; v += vmax > 20 ? 10 : 2) s += `<line x1="${x0}" x2="${x0 + w}" y1="${Y(v)}" y2="${Y(v)}" stroke="#eee"/><text x="${x0 - 6}" y="${Y(v) + 4}" font-size="10" text-anchor="end" font-family="sans-serif">${v}</text>`;
      for (const [rows, key, col] of series) {
        const pts = rows.filter((r) => r.t >= tmin && r.t <= tmax).map((r) => `${X(r.t).toFixed(1)},${Y(Math.abs(r[key])).toFixed(1)}`).join(" ");
        s += `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2"/>`;
      }
      s += `<text x="${x0 + w}" y="${y0 + h + 16}" font-size="10" text-anchor="end" font-family="sans-serif">t (s), ${tmin} .. ${tmax}</text>`;
      return s;
    };
    let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}"><rect width="${W}" height="${H}" fill="#f7f7f7"/>`;
    s += panel(pad + 10, 40, 520, 280, [[c.skate, "v", "#1f6fd1"]], 0, 2.8, 14, "1. skate 1.2 s from rest, release, coast (proportional decay)", "m/s");
    s += panel(640, 40, 520, 280, [[c.dashRest, "v", "#d1561f"], [c.dashTop, "v", "#2a9d4b"]], -0.1, 0.6, 16, "2. dash: from rest (orange), at top speed (green); Shift at t = 0", "m/s");
    s += panel(pad + 10, 420, 1110, 280, [[c.flick, "w", "#7b3fd1"]], -0.02, 0.2, 30, "3. stick flick, |omega|: up at stick_max_ang_accel, cap 26.17, stop at stick_max_ang_decel", "rad/s");
    return s + "</svg>";
  }, curves);
  const p2 = await browser.newPage({ viewport: { width: 1200, height: 760 } });
  await p2.setContent(`<html><body style="margin:0">${svg}</body></html>`);
  await p2.screenshot({ path: path.join(out, "s10-curves.png") });
  console.log(`\nscreenshots: ${path.join(out, "s10-lab.png")}, ${path.join(out, "s10-curves.png")}`);
} finally {
  await browser?.close();
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], { stdio: "ignore" });
  else { try { process.kill(-server.pid, "SIGTERM"); } catch { server.kill(); } }
}
console.log(failed ? "\nVERIFY S10 FAILED" : "\nVERIFY S10 OK");
process.exit(failed ? 1 : 0);
