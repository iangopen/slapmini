// w5b checks in the real game page with real keys: the F2 tuning panel changes the running match (player_accel, board
// restitution), Export / Reset / Import, edits carried to the lab, and Space as a slow walk (game and lab).
// Run: npm run verify-w5b (own Vite dev server). Screenshots: web/verify-out/w5b-*.png.
import { freePort } from "./free_port.mjs";
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.join(web, "verify-out");
mkdirSync(out, { recursive: true });
const PORT = await freePort();
const server = spawn(`npx vite --port ${PORT} --strictPort`, { cwd: web, shell: true, stdio: ["ignore", "pipe", "pipe"], detached: process.platform !== "win32" });
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite did not start")), 30000);
  server.stdout.on("data", (d) => { if (String(d).includes("Local:")) { clearTimeout(t); res(); } });
});
let failed = false;
const check = (c, m) => { console.log(`${c ? "PASS" : "FAIL"}: ${m}`); if (!c) failed = true; };
const launch = async () => { try { return await chromium.launch(); } catch (e) { const a = process.env.PW_CHROMIUM || "/opt/pw-browsers/chromium"; if (!existsSync(a)) throw e; return chromium.launch({ executablePath: a }); } };
const FILE = JSON.parse(readFileSync(path.join(web, "..", "params.json"), "utf8")).params;
const V = (n) => FILE[n].value;
const browser = await launch();
const ctx = await browser.newContext({ viewport: { width: 1400, height: 760 }, acceptDownloads: true });

// Space table, shared by the game and the lab: `api` reads the player and places it; real keys throughout.
async function spaceChecks(page, tag, place, read) {
  const speed = (r) => Math.hypot(r.vx, r.vy);
  const startTrace = () => page.evaluate((f) => {
    const get = eval(`(${f})`);
    window.__tr = [];
    window.__on = true;
    const loop = () => { if (!window.__on) return; window.__tr.push(get()); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }, read.toString());
  const stopTrace = () => page.evaluate(() => { window.__on = false; return window.__tr; });
  const table = (rows, t0tick, every = 0.2) => {
    let next = 0;
    console.log("   t (s) | speed (m/s) | x (m)");
    for (const q of rows) {
      const t = (q.tick - t0tick) / 120;
      if (t >= next) { console.log(`  ${t.toFixed(3).padStart(6)} | ${speed(q).toFixed(3).padStart(11)} | ${q.x.toFixed(2)}`); next = t + every; }
    }
  };
  const slow = V("brake_speed"), top = V("player_max_speed");

  await place(-28);
  await page.keyboard.down("KeyW");
  await page.waitForTimeout(1000);
  await startTrace();
  await page.waitForTimeout(100);
  await page.keyboard.down("Space");
  await page.waitForTimeout(1900);
  if (tag === "game") await page.screenshot({ path: path.join(out, "w5b-slow-badge.png") });
  let r = await stopTrace();
  console.log(`[${tag}] W held from rest, then Space (W still held), real keys:`);
  table(r, r[0].tick);
  const tail = r.slice(-40);
  const moved = tail.at(-1).x - tail[0].x, dtT = (tail.at(-1).tick - tail[0].tick) / 120;
  check(Math.abs(speed(r[0]) - top) < 1e-6, `[${tag}] at top speed before Space (${speed(r[0]).toFixed(3)} m/s)`);
  check(tail.every((q) => Math.abs(speed(q) - slow) < 1e-6), `[${tag}] settles at brake_speed ${slow} m/s, not 0`);
  check(Math.abs(moved / dtT - slow) < 0.01, `[${tag}] still moving with W held: ${moved.toFixed(3)} m in ${dtT.toFixed(3)} s`);
  // Release: normal acceleration back to top speed.
  await startTrace();
  await page.keyboard.up("Space");
  await page.waitForTimeout(900);
  r = await stopTrace();
  await page.keyboard.up("KeyW");
  const i0 = r.findIndex((q) => speed(q) > slow + 0.05);
  const i1 = r.findIndex((q) => speed(q) > slow + 6);
  const acc = 6 / ((r[i1].tick - r[Math.max(0, i0 - 1)].tick) / 120);
  check(Math.abs(speed(r.at(-1)) - top) < 1e-6, `[${tag}] Space released, W held: back at top speed ${speed(r.at(-1)).toFixed(3)} m/s`);
  check(Math.abs(acc - V("player_accel")) < 2.5, `[${tag}] release gives normal acceleration (${acc.toFixed(1)} m/s^2, player_accel ${V("player_accel")})`);

  // Space with no keys: decays to rest.
  await place(-28);
  await page.keyboard.down("KeyW");
  await page.waitForTimeout(1000);
  await page.keyboard.up("KeyW");
  await startTrace();
  await page.keyboard.down("Space");
  await page.waitForTimeout(3600);
  r = await stopTrace();
  await page.keyboard.up("Space");
  console.log(`[${tag}] Space with no keys:`);
  table(r, r[0].tick, 0.5);
  check(speed(r.at(-1)) < 0.1 && r.every((q, i) => i === 0 || speed(q) <= speed(r[i - 1]) + 1e-9), `[${tag}] decays to rest without speeding up (${speed(r.at(-1)).toFixed(3)} m/s after ${((r.at(-1).tick - r[0].tick) / 120).toFixed(2)} s)`);

  // Dash while braking.
  await place(-28);
  await page.waitForTimeout(100);
  await page.keyboard.down("Space");
  await page.keyboard.down("KeyW");
  await page.waitForTimeout(500);
  await startTrace();
  await page.waitForTimeout(80); // the first sample must be before the dash
  await page.keyboard.down("ShiftLeft");
  await page.waitForTimeout(700);
  r = await stopTrace();
  await page.keyboard.up("ShiftLeft");
  await page.keyboard.up("KeyW");
  await page.keyboard.up("Space");
  const peak = Math.max(...r.map(speed));
  console.log(`[${tag}] dash while braking: before ${speed(r[0]).toFixed(3)}, peak ${peak.toFixed(3)}, end ${speed(r.at(-1)).toFixed(3)} m/s`);
  check(Math.abs(speed(r[0]) - slow) < 1e-6 && Math.abs(peak - V("dash_speed")) < 1.0 && Math.abs(speed(r.at(-1)) - slow) < 1e-6, `[${tag}] the dash fires at full strength and the slow walk takes over after it`);
}

try {
  const page = await ctx.newPage();
  page.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  const ev = (fn, a) => page.evaluate(fn, a);
  await page.goto(`http://localhost:${PORT}/index.html?camera=fit`);
  await page.waitForSelector("body[data-ready='1']");
  const idle = await ev(() => window.game.profiles.find((p) => p.scripted === "idle")?.id ?? null);
  if (idle) await page.selectOption("#opponent", idle);
  await page.click("#play");
  await page.waitForFunction(() => window.game.screen === "match");
  const play = () => ev(() => {
    const s = structuredClone(window.game.state);
    s.match_state = { ...s.match_state, phase: "playing", phase_ticks: 0, clock_ticks: 30000 };
    window.game.setState(s);
  });
  await play();
  const place = (x) => ev((px) => {
    const s = structuredClone(window.game.state);
    Object.assign(s.players[0], { x: px, y: 14, vx: 0, vy: 0, dash_timer: 0, dash_cooldown: 0 });
    Object.assign(s.players[1], { x: 20, y: 18, vx: 0, vy: 0 });
    Object.assign(s.puck, { x: 0, y: -18, z: 0, vx: 0, vy: 0, vz: 0, in_net: null });
    window.game.setState(s);
  }, x);

  // ---- panel ----
  check(await page.locator("#tune").isHidden(), "tuning panel hidden by default");
  await page.keyboard.press("F2");
  check(await page.locator("#tune").isVisible(), "F2 opens the panel during the match");
  const rows = await page.locator("#tune .param").count();
  check(rows === Object.keys(FILE).length, `panel lists every parameter (${rows})`);
  const tick0 = await ev(() => window.game.state.tick);

  const timeTo8 = async () => {
    await place(-28);
    await ev(() => {
      window.__t = null;
      const t0 = window.game.state.tick;
      const loop = () => {
        const p = window.game.state.players[0];
        if (Math.hypot(p.vx, p.vy) >= 8) { window.__t = window.game.state; window.__t8 = window.__first ? (window.game.state.tick - window.__first) / 120 : 0; return; }
        if (!window.__first && Math.hypot(p.vx, p.vy) > 0) window.__first = window.game.state.tick;
        requestAnimationFrame(loop);
      };
      window.__first = 0;
      requestAnimationFrame(loop);
    });
    await page.keyboard.down("KeyW");
    await page.waitForFunction(() => window.__t !== null, null, { timeout: 8000 });
    await page.keyboard.up("KeyW");
    return ev(() => window.__t8);
  };
  const a0 = await timeTo8();
  await page.fill("#param-player_accel", "8");
  await page.locator("#param-player_accel").blur();
  const a1 = await timeTo8();
  console.log(`[panel] rest to 8 m/s in the running match: default accel ${a0.toFixed(3)} s; player_accel 8 through the DOM ${a1.toFixed(3)} s`);
  check(Math.abs(a0 - 8 / V("player_accel")) < 0.08 && Math.abs(a1 - 1.0) < 0.1, "player_accel changes the running match");
  const row = page.locator(".param", { has: page.locator("#param-player_accel") });
  check(await row.evaluate((e) => e.classList.contains("changed")), "changed row is marked");
  check((await row.locator(".badge").textContent()) === FILE.player_accel.provenance, `row keeps its provenance tag (${FILE.player_accel.provenance})`);

  const bounce = async () => {
    await ev(() => {
      const s = structuredClone(window.game.state);
      Object.assign(s.players[0], { x: -28, y: 0, vx: 0, vy: 0 });
      Object.assign(s.puck, { x: 5, y: 16, z: 0, vx: 0, vy: 15, vz: 0, in_net: null });
      window.game.setState(s);
      window.__b = null; window.__vin = 0;
      const loop = () => {
        const k = window.game.state.puck;
        if (k.vy > 0) window.__vin = k.vy;
        if (k.vy < 0) { window.__b = -k.vy / window.__vin; return; }
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    });
    await page.waitForFunction(() => window.__b !== null, null, { timeout: 5000 });
    return ev(() => window.__b);
  };
  const e0 = await bounce();
  await page.fill("#param-board_restitution", "0.9");
  await page.locator("#param-board_restitution").blur();
  const e1 = await bounce();
  console.log(`[panel] puck off the board in the running match: rebound/incoming ${e0.toFixed(3)} at the default ${V("board_restitution")}; ${e1.toFixed(3)} with board_restitution 0.9 through the DOM`);
  check(Math.abs(e0 - V("board_restitution")) < 0.06 && Math.abs(e1 - 0.9) < 0.1, "board_restitution changes the running match");
  const live = await ev(() => ({ tick: window.game.state.tick, started: window.game.matchesStarted, phase: window.game.state.match_state.phase, tuned: document.getElementById("tuned").hidden }));
  check(live.tick > tick0 && live.started === 1 && live.phase === "playing", `same match kept running (tick ${tick0} -> ${live.tick}, matches started ${live.started})`);
  await page.screenshot({ path: path.join(out, "w5b-panel.png") });

  // ---- export, reset, import ----
  const [dl] = await Promise.all([page.waitForEvent("download"), page.click("#params-export")]);
  const exported = path.join(out, "w5b-export.json");
  await dl.saveAs(exported);
  const ex = JSON.parse(readFileSync(exported, "utf8")).params;
  check(ex.player_accel.value === 8 && ex.board_restitution.value === 0.9 && ex.player_accel.provenance === FILE.player_accel.provenance, `exported JSON holds the edits (accel ${ex.player_accel.value}, restitution ${ex.board_restitution.value}) and the tags`);
  await page.click("#params-reset");
  const reset = await ev(() => ({ a: window.game.params.player_accel, n: document.querySelectorAll("#tune .param.changed").length, v: document.getElementById("param-player_accel").value }));
  check(reset.a === V("player_accel") && reset.n === 0 && +reset.v === V("player_accel"), `Reset restores params.json (accel ${reset.a}, ${reset.n} changed rows)`);
  await page.setInputFiles("#params-import-file", exported);
  await page.waitForFunction(() => window.game.params.player_accel === 8);
  const a2 = await timeTo8();
  const e2 = await bounce();
  console.log(`[panel] after Import: rest to 8 m/s ${a2.toFixed(3)} s, rebound/incoming ${e2.toFixed(3)}; status: ${await page.textContent("#params-status")}`);
  check(Math.abs(a2 - 1.0) < 0.1 && Math.abs(e2 - 0.9) < 0.1 && (await page.locator("#tune .param.changed").count()) === 2, "Import restores the edits in the running match");
  const bad = path.join(out, "w5b-bad.json");
  writeFileSync(bad, JSON.stringify({ params: { nonsense: { value: 1 } } }));
  await page.setInputFiles("#params-import-file", bad);
  await page.waitForFunction(() => document.getElementById("params-status").textContent.includes("refused"));
  check((await ev(() => window.game.params.player_accel)) === 8, `bad import refused: ${await page.textContent("#params-status")}`);

  // ---- an edit reaches the other page ----
  const lab = await ctx.newPage();
  await lab.goto(`http://localhost:${PORT}/lab.html?mouse=cursor`);
  await lab.waitForSelector("body[data-ready='1']");
  const carried = await lab.evaluate(() => ({ p: window.lab.lab.p.player_accel, field: document.getElementById("param-player_accel").value, changed: document.querySelectorAll("#params .param.changed").length }));
  check(carried.p === 8 && +carried.field === 8 && carried.changed === 2, `lab opens with the game's edits (accel ${carried.p}, ${carried.changed} changed rows)`);
  await lab.click("#params-reset");
  await lab.close();
  await page.click("#params-reset");
  await page.keyboard.press("F2");
  check(await page.locator("#tune").isHidden(), "F2 closes the panel");

  // ---- Space ----
  await play();
  await spaceChecks(page, "game", place, () => { const s = window.game.state, p = s.players[0]; return { tick: s.tick, x: p.x, vx: p.vx, vy: p.vy }; });
  const inputs = await ev(() => window.game.lastInputs[0]);
  check("brake" in inputs, "the game sends PlayerInput.brake");

  const lp = await ctx.newPage();
  lp.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  await lp.goto(`http://localhost:${PORT}/lab.html?mouse=cursor`);
  await lp.waitForSelector("body[data-ready='1']");
  await lp.mouse.click(300, 300);
  const placeLab = (x) => lp.evaluate((px) => {
    const s = structuredClone(window.lab.lab.state);
    Object.assign(s.players[0], { x: px, y: 14, vx: 0, vy: 0, dash_timer: 0, dash_cooldown: 0 });
    Object.assign(s.players[1], { x: 20, y: 18, vx: 0, vy: 0 });
    window.lab.lab.setState(s);
  }, x);
  await spaceChecks(lp, "lab", placeLab, () => { const s = window.lab.lab.state, p = s.players[0]; return { tick: s.tick, x: p.x, vx: p.vx, vy: p.vy }; });
} finally {
  await browser.close();
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], { stdio: "ignore" });
  else { try { process.kill(-server.pid, "SIGTERM"); } catch { server.kill(); } }
}
console.log(failed ? "\nVERIFY W5B FAILED" : "\nVERIFY W5B OK");
process.exit(failed ? 1 : 0);
