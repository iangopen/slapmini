// w2 checks: the vertical view (game + lab), the relative mouse (pointer lock), Esc, and record/replay
// in relative mode. Run: npm run verify-w2 (own Vite dev server). Screenshots in web/verify-out/w2-*.png.
import { freePort } from "./free_port.mjs";
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
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
const deg = (r) => ((r * 180) / Math.PI).toFixed(2);
const launch = async () => { try { return await chromium.launch(); } catch (e) { const a = process.env.PW_CHROMIUM || "/opt/pw-browsers/chromium"; if (!existsSync(a)) throw e; return chromium.launch({ executablePath: a }); } };
const browser = await launch();
try {
  // ================= game: vertical view =================
  const page = await browser.newPage({ viewport: { width: 1280, height: 760 } });
  page.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  const ev = (fn, a) => page.evaluate(fn, a);
  await page.goto(`http://localhost:${PORT}/`);
  await page.waitForSelector("body[data-ready='1']");
  await page.screenshot({ path: path.join(out, "w2-menu.png") });
  await page.selectOption("#opponent", "defender");
  await page.click("#play");
  await page.waitForFunction(() => window.game.screen === "match" && window.game.summary.phase === "playing", null, { timeout: 15000 });
  const netX = await ev(() => window.game.params.net_x);
  const top = await ev((x) => window.game.worldToClient(x, 0), netX);
  const bottom = await ev((x) => window.game.worldToClient(-x, 0), netX);
  console.log(`[view] +x net (BLUE attacks) at client y ${top[1].toFixed(0)}, −x net (BLUE defends) at y ${bottom[1].toFixed(0)}`);
  check(bottom[1] > top[1], "BLUE's own goal (−x) is at the bottom, the goal it attacks at the top");
  // Each key changes the player's VELOCITY in its screen direction (positions would mix in the momentum
  // left over from the previous key). Screen direction of a world velocity: worldToClient(p + v) − worldToClient(p).
  const me = () => ev(() => { const p = window.game.state.players[0]; const a = window.game.worldToClient(p.x, p.y), b = window.game.worldToClient(p.x + p.vx, p.y + p.vy); return { x: p.x, y: p.y, vx: p.vx, vy: p.vy, sv: [b[0] - a[0], b[1] - a[1]] }; });
  for (const [key, want] of [["KeyW", "up"], ["KeyA", "left"], ["KeyS", "down"], ["KeyD", "right"]]) {
    const a = await me();
    await page.keyboard.down(key);
    await page.waitForTimeout(300);
    const inp = await ev(() => window.game.lastInputs[0]);
    await page.keyboard.up(key);
    const b = await me();
    const dsx = b.sv[0] - a.sv[0], dsy = b.sv[1] - a.sv[1]; // change of the on-screen velocity (px per s)
    const dvx = b.vx - a.vx, dvy = b.vy - a.vy;
    console.log(`[keys] ${key}: sim got move_x ${inp.move_x} move_y ${inp.move_y}; velocity change on screen (${dsx.toFixed(0)}, ${dsy.toFixed(0)}) px/s, in the world (${dvx.toFixed(2)}, ${dvy.toFixed(2)}) m/s`);
    const ok = {
      up: dsy < -20 && Math.abs(dsx) < Math.abs(dsy) / 3 && dvx > 1 && inp.move_x === 1 && inp.move_y === 0,
      left: dsx < -20 && Math.abs(dsy) < Math.abs(dsx) / 3 && dvy > 1 && inp.move_x === 0 && inp.move_y === 1,
      down: dsy > 20 && Math.abs(dsx) < Math.abs(dsy) / 3 && dvx < -1 && inp.move_x === -1 && inp.move_y === 0,
      right: dsx > 20 && Math.abs(dsy) < Math.abs(dsx) / 3 && dvy < -1 && inp.move_x === 0 && inp.move_y === -1,
    }[want];
    check(ok, `${key} pushes the player ${want} the screen, and the world change matches (+x up, +y left)`);
  }
  await page.screenshot({ path: path.join(out, "w2-midmatch.png") });

  // ================= game: relative mouse =================
  let aim = await ev(() => window.game.aim);
  console.log(`[mouse] mode ${aim.mode}, sensitivity ${aim.sensitivity}, hint shown ${aim.hint}, locked ${aim.locked}; stick screen angle ${deg(aim.screenAngle)}° (forward = up = 90°), world ${deg(aim.worldAngle)}°`);
  check(aim.mode === "relative" && aim.hint && !aim.locked, "relative mode by default; the 'click to capture the mouse' hint shows before capture");
  check(Math.abs(aim.screenAngle - Math.PI / 2) < 1e-9 && Math.abs(aim.worldAngle) < 1e-9, "the stick starts pointing forward (screen up = world +x, toward the opponent's goal)");
  await page.mouse.move(640, 380);
  await page.mouse.click(640, 380); // the first click in a match captures the pointer
  await page.waitForTimeout(200);
  aim = await ev(() => window.game.aim);
  const realLock = aim.locked;
  console.log(`[mouse] after clicking the rink: pointer locked = ${realLock} (Pointer Lock API in headless Chromium), hint shown ${aim.hint}`);
  // Move the mouse by (dx, dy): with a real pointer lock, Playwright's mouse.move produces movementX/Y
  // (then moves back, which is the inverse turn); without one, the test hook calls the same handler.
  async function delta(dx, dy) {
    const before = await ev(() => window.game.aim);
    if (realLock) {
      await page.mouse.move(640 + dx, 380 + dy, { steps: 8 });
      const mid = await ev(() => window.game.aim);
      await page.mouse.move(640, 380, { steps: 8 }); // back: the inverse turn
      return { before, after: mid };
    }
    await ev(([x, y]) => window.game.mouseDelta(x, y), [dx, dy]);
    const after = await ev(() => window.game.aim);
    await ev(([x, y]) => window.game.mouseDelta(-x, -y), [dx, dy]);
    return { before, after };
  }
  const sens = aim.sensitivity;
  const N = 100;
  for (const [dx, dy, what] of [[N, 0, "right"], [-N, 0, "left"], [0, N, "down (vertical)"], [0, -N, "up (vertical)"]]) {
    const { before, after } = await delta(dx, dy);
    const d = after.screenAngle - before.screenAngle;
    const want = -dx * sens;
    console.log(`[mouse] move ${what} by ${N} (${realLock ? "real pointer-lock movementX" : "test hook"}): screen angle ${deg(before.screenAngle)}° → ${deg(after.screenAngle)}° (Δ ${deg(d)}°, expected ${deg(want)}°); world stick target ${deg(after.worldAngle)}°`);
    if (dx > 0) check(Math.abs(d - want) < 1e-6 && d < 0, `moving right by ${N} turns the stick clockwise on screen (to the right) by ${deg(-want)}°`);
    if (dx < 0) check(Math.abs(d - want) < 1e-6 && d > 0, `moving left by ${N} turns the stick counterclockwise on screen (to the left) by ${deg(want)}°`);
    if (dx === 0) check(Math.abs(d) < 1e-12, "vertical mouse movement changes nothing");
  }
  // The sim receives the world angle of the screen aim (through the view transform): turned right from up = toward −y (screen right).
  await ev(() => window.game.mouseDelta(100, 0));
  await page.waitForTimeout(100);
  const sent = await ev(() => ({ inp: window.game.lastInputs[0].stick_target_angle, aim: window.game.aim }));
  await ev(() => window.game.mouseDelta(-100, 0));
  console.log(`[mouse] after turning right: sim stick_target_angle ${deg(sent.inp)}° (world), aim ${deg(sent.aim.worldAngle)}°`);
  check(Math.abs(sent.inp - sent.aim.worldAngle) < 1e-9 && sent.inp < 0, "the sim gets the aim as a world angle; right of forward = toward −y (screen right in the vertical view)");
  // Sensitivity slider scales the turn.
  await ev(() => window.game.setSensitivity(0.01));
  const s2 = await delta(N, 0);
  const d2 = s2.after.screenAngle - s2.before.screenAngle;
  console.log(`[mouse] sensitivity ${sens} → 0.01: moving right by ${N} turns ${deg(-d2)}° (was ${deg(N * sens)}°)`);
  check(Math.abs(d2 + N * 0.01) < 1e-6, "the sensitivity slider scales the turn (0.01 rad per unit)");
  const stored = await ev(() => { try { return JSON.parse(localStorage.getItem("slapmini.mouse.v1")); } catch { return null; } });
  check(stored && Math.abs(stored.sensitivity - 0.01) < 1e-12, `the setting is stored in localStorage (${JSON.stringify(stored)})`);
  await ev(() => window.game.setSensitivity(0.005));
  // Esc: the first releases the mouse (match continues), the second leaves.
  if (realLock) {
    await page.keyboard.press("Escape");
    await page.waitForTimeout(150);
    let s = await ev(() => ({ screen: window.game.screen, aim: window.game.aim, phase: window.game.summary.phase }));
    console.log(`[esc] after Esc: locked ${s.aim.locked}, screen ${s.screen}, phase ${s.phase}, hint ${s.aim.hint}`);
    check(!s.aim.locked && s.screen === "match" && s.aim.hint, "Esc releases the mouse, the match goes on, the capture hint is back");
    await page.waitForTimeout(400);
  }
  await page.keyboard.press("Escape");
  await page.waitForTimeout(150);
  check((await ev(() => window.game.screen)) === "menu", "Esc (without a captured mouse) returns to the menu");

  // ================= lab: vertical view, F spawn at the clicked pixel, old view =================
  const lab = await browser.newPage({ viewport: { width: 1400, height: 760 } });
  lab.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  const lev = (fn, a) => lab.evaluate(fn, a);
  await lab.goto(`http://localhost:${PORT}/lab.html`);
  await lab.waitForSelector("body[data-ready='1']");
  check(await lab.isChecked("#game-view"), "lab: 'Rink vertical' is on by default");
  const la = await lev(() => window.lab.aim);
  console.log(`[lab] initial aim: screen ${deg(la.screenAngle)}°, world ${deg(la.worldAngle)}°`);
  check(Math.abs(la.screenAngle - Math.PI / 2) < 1e-9 && Math.abs(la.worldAngle) < 1e-9, "lab: the relative aim starts forward (up the screen = world +x)");
  const pix = [480, 260];
  await lab.mouse.move(pix[0], pix[1]);
  await lab.keyboard.press("KeyF");
  await lab.waitForTimeout(100);
  const sp = await lev(() => window.lab.lab.spawn);
  const back = await lev(([x, y]) => window.lab.worldToClient(x, y), [sp.x, sp.y]);
  console.log(`[lab] F at client (${pix}) → spawn world (${sp.x.toFixed(3)}, ${sp.y.toFixed(3)}) → back to client (${back[0].toFixed(2)}, ${back[1].toFixed(2)})`);
  check(Math.hypot(back[0] - pix[0], back[1] - pix[1]) < 1, "lab: F sets the spawn at the clicked pixel through the vertical transform");
  await lab.keyboard.press("KeyR");
  await lab.screenshot({ path: path.join(out, "w2-lab-vertical.png") });
  await lab.uncheck("#game-view");
  await lab.waitForTimeout(150);
  await lab.mouse.move(pix[0], pix[1]);
  await lab.keyboard.press("KeyF");
  await lab.waitForTimeout(100);
  const sp2 = await lev(() => window.lab.lab.spawn);
  const back2 = await lev(([x, y]) => window.lab.worldToClient(x, y), [sp2.x, sp2.y]);
  console.log(`[lab] old horizontal view: F at (${pix}) → spawn (${sp2.x.toFixed(3)}, ${sp2.y.toFixed(3)}) → client (${back2[0].toFixed(2)}, ${back2[1].toFixed(2)})`);
  check(Math.hypot(back2[0] - pix[0], back2[1] - pix[1]) < 1 && (Math.abs(sp2.x - sp.x) > 1 || Math.abs(sp2.y - sp.y) > 1), "lab: the old horizontal view is still reachable and its clicks map correctly too");
  await lab.keyboard.press("KeyR");
  await lab.screenshot({ path: path.join(out, "w2-lab-horizontal.png") });
  await lab.check("#game-view");

  // ================= lab: record + replay in relative mouse mode =================
  await lev(() => window.lab.setMouseMode("relative"));
  await lab.click("#rec"); // start recording first: while the mouse is captured, panel clicks go to the canvas
  await lab.mouse.click(700, 380); // capture the mouse
  console.log(`[lab] relative mode, mouse captured: ${(await lev(() => window.lab.aim)).locked}`);
  await lab.keyboard.down("KeyW");
  for (let i = 0; i < 6; i++) {
    await lev((d) => window.lab.mouseDelta(d, 0), i % 2 ? -60 : 90);
    await lab.waitForTimeout(120);
  }
  await lab.keyboard.down("KeyE"); // lift (was Space before w3)
  await lab.waitForTimeout(200);
  await lab.keyboard.up("KeyE");
  await lab.keyboard.up("KeyW");
  await lab.waitForTimeout(200);
  await lab.keyboard.press("Escape"); // release the mouse, then the panel is clickable again
  check(!(await lev(() => window.lab.aim)).locked, "lab: Esc releases the captured mouse");
  await lab.click("#rec");
  const rec = await lev(() => ({ csv: window.lab.toCsv(window.lab.lab.lastRecording.samples), steps: window.lab.lab.lastRecording.scenario.steps, events: window.lab.lab.lastRecording.scenario.inputs.length,
    lifts: window.lab.lab.lastRecording.scenario.inputs.filter((e) => e.input.lift).length, angles: new Set(window.lab.lab.lastRecording.scenario.inputs.map((e) => e.input.stick_target_angle.toFixed(4))).size }));
  await lab.click("#replay");
  await lab.waitForFunction(() => window.lab.lab.mode === "play" && window.lab.lab.lastReplay, null, { timeout: 30000 });
  const rep = await lev(() => window.lab.toCsv(window.lab.lab.lastReplay.samples));
  const h = (s) => createHash("sha256").update(s).digest("hex");
  console.log(`[record/replay, relative mouse] ${rec.steps} steps, ${rec.events} input events (${rec.angles} distinct stick angles, ${rec.lifts} with lift)\n  recorded sha256 ${h(rec.csv)}\n  replayed sha256 ${h(rep)}`);
  check(h(rec.csv) === h(rep) && rec.angles > 3 && rec.lifts > 0, "recording with the relative mouse replays byte-identically (it logs stick_target_angle, not mouse events)");
} finally {
  await browser.close();
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], { stdio: "ignore" });
  else { try { process.kill(-server.pid, "SIGTERM"); } catch { server.kill(); } }
}
console.log(failed ? "\nVERIFY W2 FAILED" : "\nVERIFY W2 OK");
process.exit(failed ? 1 : 0);
