// w4 check with REAL mouse buttons: the raised stick (left button) in the game and the lab, the follow camera and
// true-scale radii. Run: npm run wasm && npm run verify-w4   (own Vite dev server, VERIFY_PORT or 5191)
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.join(web, "verify-out");
mkdirSync(out, { recursive: true });
const PORT = Number(process.env.VERIFY_PORT) || 5191;
const server = spawn(`npx vite --port ${PORT} --strictPort`, { cwd: web, shell: true, stdio: ["ignore", "pipe", "pipe"], detached: process.platform !== "win32" });
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite did not start")), 30000);
  server.stdout.on("data", (d) => { if (String(d).includes("Local:")) { clearTimeout(t); res(); } });
});
const sha = (s) => createHash("sha256").update(s).digest("hex");
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
  // ---------------- game: relative mode (the default) ----------------
  const page = await browser.newPage({ viewport: { width: 1280, height: 760 } });
  page.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  await page.goto(`http://localhost:${PORT}/`);
  await page.waitForSelector("body[data-ready='1']");
  await page.selectOption("#opponent", "idle").catch(() => page.selectOption("#opponent", { index: 0 }));
  await page.click("#play");
  await page.waitForFunction(() => window.game.screen === "match" && window.game.summary.phase === "playing", null, { timeout: 15000 });
  const g = () => page.evaluate(() => ({ ...window.game.w4, locked: window.game.aim.locked, in0: window.game.lastInputs[0] ?? null }));
  let s = await g();
  console.log(`[camera] ${s.camera}, ${s.scale.toFixed(2)} px/m; drawn radii player ${s.radii.player.toFixed(2)} px, puck ${s.radii.puck.toFixed(2)} px (ratio ${(s.radii.player / s.radii.puck).toFixed(3)})`);
  check(s.camera === "follow" && Math.abs(s.radii.player / s.radii.puck - 0.4969 / 0.249853) < 1e-6, "follow camera by default; player and puck drawn at true scale");
  const box = await page.locator("#rink").boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  // 1. the capturing click must not raise the stick
  await page.mouse.down();
  await page.waitForTimeout(120);
  s = await g();
  console.log(`[capture click] locked ${s.locked}, stickUp ${s.stickUp}, sim input stick_up ${s.in0?.stick_up}`);
  check(s.locked, "the first click captured the pointer (real Pointer Lock)");
  check(!s.stickUp && !s.in0?.stick_up, "the capturing click (held) does not raise the stick");
  await page.mouse.up();
  // 2. held left button raises
  await page.mouse.down();
  await page.waitForTimeout(150);
  s = await g();
  const st0 = await page.evaluate(() => window.game.state.players[0].stick_up);
  console.log(`[held] stickUp ${s.stickUp}, input stick_up ${s.in0?.stick_up}, sim player stick_up ${st0}`);
  check(s.stickUp && s.in0?.stick_up === true && st0 === true, "holding the left button raises the stick (input and sim state true)");
  await page.screenshot({ path: path.join(out, "w4-stick-up-game.png") });
  // 3. release lowers
  await page.mouse.up();
  await page.waitForTimeout(120);
  s = await g();
  check(!s.stickUp && s.in0?.stick_up === false, "releasing the left button lowers it");
  // 4. Esc while held lowers it (pointer unlock)
  await page.mouse.down();
  await page.waitForTimeout(80);
  await page.keyboard.press("Escape");
  await page.waitForTimeout(150);
  s = await g();
  console.log(`[esc while held] locked ${s.locked}, stickUp ${s.stickUp}, input ${s.in0?.stick_up}`);
  check(!s.locked && !s.stickUp && s.in0?.stick_up === false, "Esc (pointer unlock) lowers a held stick");
  await page.mouse.up();
  // 5. whole-rink camera still available
  await page.evaluate(() => window.game.setCamera("fit"));
  s = await g();
  check(s.camera === "fit" && s.scale < 15, `whole-rink camera available (${s.scale.toFixed(2)} px/m)`);
  await page.close();

  // ---------------- lab: cursor mode, puck rolled across the stick ----------------
  const lab = await browser.newPage({ viewport: { width: 1400, height: 760 } });
  lab.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  await lab.goto(`http://localhost:${PORT}/lab.html?mouse=cursor`);
  await lab.waitForSelector("body[data-ready='1']");
  const w2c = (x, y) => lab.evaluate(([x, y]) => window.lab.worldToClient(x, y), [x, y]);
  const roll = async (up) => {
    // player at rest at the origin, stick pointing along +x at the cursor; puck rolls along -y across the blade
    await lab.evaluate(() => {
      const s = structuredClone(window.lab.lab.state);
      Object.assign(s.players[0], { x: 0, y: 0, vx: 0, vy: 0, stick_angle: 0, stick_omega: 0 });
      Object.assign(s.players[1], { x: 20, y: 15, vx: 0, vy: 0 });
      Object.assign(s.puck, { x: 50, y: 0, z: 0, vx: 0, vy: 0, vz: 0 });
      window.lab.lab.setState(s);
    });
    const [px, py] = await w2c(4, 0);
    await lab.mouse.move(px, py);
    if (up) await lab.mouse.down();
    await lab.waitForTimeout(250);
    await lab.evaluate(() => {
      const s = structuredClone(window.lab.lab.state);
      Object.assign(s.puck, { x: 0.95, y: 3, z: 0, vx: 0, vy: -6, vz: 0 });
      window.lab.lab.setState(s);
    });
    await lab.waitForTimeout(900);
    const r = await lab.evaluate(() => ({ puck: window.lab.lab.state.puck, up: !!window.lab.lab.state.players[0].stick_up, inp: !!window.lab.lab.inputs[0].stick_up }));
    if (up) { await lab.screenshot({ path: path.join(out, "w4-stick-up-lab.png") }); await lab.mouse.up(); }
    return r;
  };
  const down = await roll(false);
  const upr = await roll(true);
  const fmt = (p) => `x ${p.x.toFixed(2)} y ${p.y.toFixed(2)} vx ${p.vx.toFixed(2)} vy ${p.vy.toFixed(2)}`;
  console.log(`[lab roll] stick down: ${fmt(down.puck)}`);
  console.log(`[lab roll] stick up (held, sim stick_up ${upr.up}): ${fmt(upr.puck)}`);
  check(upr.up && upr.inp, "lab: holding the left button raises the stick (cursor mode, no capture)");
  check(Math.abs(down.puck.vx) > 0.5 || down.puck.vy > -5, "stick down: the rolling puck is deflected by the blade");
  check(Math.abs(upr.puck.vx) < 1e-9 && upr.puck.y < -1 && Math.abs(upr.puck.x - 0.95) < 1e-9, "stick up: the puck passes under it untouched");

  // record -> replay with raised sticks
  await lab.evaluate(() => {
    const s = structuredClone(window.lab.lab.state);
    Object.assign(s.players[0], { x: -8, y: -2, vx: 0, vy: 0 });
    Object.assign(s.puck, { x: -6.6, y: -2, z: 0, vx: 0, vy: 0, vz: 0 });
    window.lab.lab.setState(s);
  });
  const mv = async (x, y) => { const [a, b] = await w2c(x, y); await lab.mouse.move(a, b); };
  await mv(-8, -6);
  await lab.click("#rec");
  await mv(-4, 1);
  await lab.keyboard.down("KeyD");
  await lab.mouse.down();
  await lab.waitForTimeout(300);
  await mv(-2, 4);
  await lab.mouse.up();
  await lab.waitForTimeout(250);
  await lab.mouse.down();
  await mv(-10, -4);
  await lab.waitForTimeout(300);
  await lab.mouse.up();
  await lab.keyboard.up("KeyD");
  await lab.waitForTimeout(300);
  await lab.click("#rec");
  const rec = await lab.evaluate(() => ({ csv: window.lab.toCsv(window.lab.lab.lastRecording.samples), sc: window.lab.lab.lastRecording.scenario, hash: window.lab.lab.lastRecording.endHash }));
  const ups = rec.sc.inputs.filter((e) => e.input.stick_up).length;
  await lab.evaluate(() => (window.lab.lab.lastReplay = null));
  await lab.click("#replay");
  await lab.waitForFunction(() => window.lab.lab.mode === "play" && window.lab.lab.lastReplay, null, { timeout: 30000 });
  const rep = await lab.evaluate(() => ({ csv: window.lab.toCsv(window.lab.lab.lastReplay.samples), hash: window.lab.lab.lastReplay.endHash }));
  writeFileSync(path.join(out, "w4_run.scenario.json"), JSON.stringify(rec.sc, null, 2) + "\n");
  console.log(`[record/replay] ${rec.sc.steps} steps, ${rec.sc.inputs.length} events (${ups} with stick_up), end hash ${rec.hash} / ${rep.hash}`);
  check(ups >= 2, "the recording carries raised-stick input events");
  check(sha(rec.csv) === sha(rep.csv) && rec.hash === rep.hash, "record -> replay with raised sticks is byte-identical (trajectory and end state hash)");
} finally {
  await browser?.close();
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], { stdio: "ignore" });
  else { try { process.kill(-server.pid, "SIGTERM"); } catch { server.kill(); } }
}
console.log(failed ? "VERIFY W4 FAILED" : "VERIFY W4 OK");
process.exit(failed ? 1 : 0);
