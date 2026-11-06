// Game verification through the real UI. Serves the BUILT site (web/dist) from a subfolder
// (/slapmini/, like GitHub Pages' /REPO/) so relative paths are tested too. Run: npm run build && npm run verify-game
// Prints score / clock / phase, bot movement and shots, inference and frame times; screenshots to web/verify-out/.
import { freePort } from "./free_port.mjs";
import { chromium } from "playwright";
import { createServer } from "node:http";
import { existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(here, "..");
const dist = path.join(web, "dist");
const out = path.join(web, "verify-out");
mkdirSync(out, { recursive: true });
if (!existsSync(path.join(dist, "index.html"))) throw new Error("web/dist missing: run npm run build first");

const SUB = "/slapmini/";
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json",
  ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".png": "image/png" };
const server = createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (!u.startsWith(SUB)) { res.writeHead(404).end("outside the subfolder: " + u); return; } // absolute paths would land here
  let f = path.join(dist, u.slice(SUB.length));
  if (existsSync(f) && statSync(f).isDirectory()) f = path.join(f, "index.html");
  if (!f.startsWith(dist) || !existsSync(f)) { res.writeHead(404).end("not found"); return; }
  res.writeHead(200, { "Content-Type": TYPES[path.extname(f)] ?? "application/octet-stream" }).end(readFileSync(f));
});
const PORT = await freePort();
await new Promise((r) => server.listen(PORT, r));
const BASE = `http://localhost:${PORT}${SUB}`;

let failed = false;
const check = (c, m) => { console.log(`${c ? "PASS" : "FAIL"}: ${m}`); if (!c) failed = true; };
const outside = [];
async function launch() {
  try { return await chromium.launch(); } catch (e) {
    const alt = process.env.PW_CHROMIUM || "/opt/pw-browsers/chromium";
    if (!existsSync(alt)) throw e;
    return chromium.launch({ executablePath: alt });
  }
}

const browser = await launch();
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 760 } });
  page.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  page.on("console", (m) => { if (m.type() === "error" && !/Could not load|policy|404|Failed to load resource/.test(m.text())) console.log("console error:", m.text()); });
  page.on("response", (r) => { if (r.status() === 404 && !r.url().includes("/bots/")) outside.push(r.url()); });
  let ortFetched = false;
  page.on("request", (r) => { if (/ort-wasm.*\.wasm/.test(r.url())) ortFetched = true; });
  const ev = (fn, a) => page.evaluate(fn, a);
  const summary = () => ev(() => window.game.summary);
  const botInfo = () => ev(() => window.game.bot);
  const line = (s) => `score ${s.score[0]}-${s.score[1]} | clock ${s.clock} | phase ${s.phase}${s.overtime ? " (OT)" : ""}${s.winner !== null ? ` | winner ${s.winner === 0 ? "BLUE (human)" : "ORANGE (bot)"}` : ""}`;

  // ---- 1. menu (served from a subfolder) ----
  await page.goto(BASE);
  await page.waitForSelector("body[data-ready='1']", { timeout: 30000 });
  const opts = await page.$$eval("#opponent option", (o) => o.map((x) => `${x.value}: ${x.textContent}`));
  console.log("[menu] opponents:", opts.join(" | "));
  check(opts.some((o) => o.startsWith("chaser")) && opts.some((o) => o.startsWith("smoke")), "menu lists the scripted bots and the smoke-test ONNX bot");
  check(await ev(() => document.activeElement?.id) === "play", "keyboard focus starts on Play vs Bot");
  await page.keyboard.press("ArrowDown");
  check(await ev(() => document.activeElement?.id) === "practice", "ArrowDown moves focus to Practice");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Enter");
  check(!(await page.isHidden("#controls")), "Enter on Controls opens the controls screen");
  console.log("[controls]", (await page.textContent("#controls .card")).replace(/\s+/g, " ").slice(0, 300));
  await page.screenshot({ path: path.join(out, "game-controls.png") });
  await page.keyboard.press("Enter"); // Back
  await page.screenshot({ path: path.join(out, "game-menu.png") });

  // ---- 2. match vs the scripted chaser, human driven by key presses ----
  await page.selectOption("#opponent", "chaser");
  await page.focus("#play");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => window.game.screen === "match" && window.game.summary.phase === "playing", null, { timeout: 15000 });
  console.log("\n[vs chaser] human: W / D / A with the mouse moving; real time");
  await page.mouse.move(640, 200);
  const human0 = await ev(() => ({ x: window.game.state.players[0].x, y: window.game.state.players[0].y }));
  for (const [key, ms] of [["KeyW", 900], ["KeyD", 500], ["KeyA", 500], ["KeyS", 400]]) {
    await page.keyboard.down(key);
    await page.mouse.move(640 + (Math.random() - 0.5) * 300, 300);
    await page.waitForTimeout(ms);
    await page.keyboard.up(key);
    console.log(`  after ${key} ${ms} ms: ${line(await summary())}`);
  }
  const human1 = await ev(() => ({ x: window.game.state.players[0].x, y: window.game.state.players[0].y }));
  check(Math.hypot(human1.x - human0.x, human1.y - human0.y) > 2, `the human player moved with the keys (${Math.hypot(human1.x - human0.x, human1.y - human0.y).toFixed(1)} m)`);
  await page.waitForTimeout(3000);
  let b = await botInfo();
  console.log(`  ${line(await summary())} | bot ${b.label}: travelled ${b.travel.toFixed(1)} m, shots ${b.shots}`);
  await page.screenshot({ path: path.join(out, "game-midmatch-chaser.png") });
  check(b.travel > 5, "the scripted bot moves");
  // Mercy: the human stops; at 8x the chaser should run up a 5-goal lead.
  await ev(() => window.game.setSpeed(8));
  await page.waitForFunction(() => window.game.summary.phase === "finished", null, { timeout: 60000 });
  b = await botInfo();
  let s = await summary();
  console.log(`  END ${line(s)} | bot shots ${b.shots}, travel ${b.travel.toFixed(0)} m`);
  check(s.winner === 1 && s.score[1] - s.score[0] >= 5, "match vs chaser ends by the MERCY rule (5-goal lead)");
  check(b.shots >= 5, `the scripted bot shoots (${b.shots} shots)`);
  check(!(await page.isHidden("#gameover")), "game-over screen shows");
  console.log("  game over:", (await page.textContent("#gameover .card")).replace(/\s+/g, " "));
  await page.screenshot({ path: path.join(out, "game-over-mercy.png") });
  // Rematch with Enter, then Esc back to the menu.
  await ev(() => window.game.setSpeed(1));
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => window.game.summary.phase !== "finished" && window.game.summary.score[1] === 0, null, { timeout: 10000 });
  check(true, `Enter = rematch: ${line(await summary())}`);
  await page.keyboard.press("Escape");
  check(await ev(() => window.game.screen) === "menu", "Esc returns to the menu");

  // ---- 3. match vs the smoke-test ONNX bot ----
  check(!ortFetched, "the onnxruntime wasm (14.8 MB) was not downloaded for scripted opponents");
  await page.selectOption("#opponent", "smoke");
  await page.click("#play");
  await page.waitForFunction(() => window.game.screen === "match" && window.game.summary.phase === "playing", null, { timeout: 30000 });
  b = await botInfo();
  console.log(`\n[vs smoke ONNX bot] bot ${b.label}${b.message ? " MESSAGE " + b.message : ""}`);
  check(b.label.startsWith("policy") && !b.message, "the ONNX policy loaded (no fallback)");
  await page.mouse.move(640, 300);
  for (const [key, ms] of [["KeyW", 700], ["KeyA", 600], ["KeyW", 500]]) {
    await page.keyboard.down(key);
    await page.waitForTimeout(ms);
    await page.keyboard.up(key);
  }
  await page.waitForTimeout(6000);
  b = await botInfo();
  s = await summary();
  console.log(`  ${line(s)} | travelled ${b.travel.toFixed(1)} m, shots ${b.shots}, decisions ${b.decisions}, skipped ${b.skipped}`);
  console.log(`  inference per decision: n ${b.inferenceMs.n}, mean ${b.inferenceMs.mean.toFixed(3)} ms, p50 ${b.inferenceMs.p50.toFixed(3)}, p95 ${b.inferenceMs.p95.toFixed(3)}, max ${b.inferenceMs.max.toFixed(3)} ms`);
  const fr = await ev(() => window.game.frames);
  console.log(`  frames: n ${fr.n}, frame interval p50 ${fr.frameMs.p50.toFixed(2)} ms p95 ${fr.frameMs.p95.toFixed(2)} ms; work per frame (sim + bot + draw) mean ${fr.workMs.mean.toFixed(3)} ms p95 ${fr.workMs.p95.toFixed(3)} ms max ${fr.workMs.max.toFixed(2)} ms`);
  await page.screenshot({ path: path.join(out, "game-midmatch-onnx.png") });
  check(b.travel > 5, "the ONNX bot moves");
  check(b.decisions > 100, `the ONNX bot decides at ~30 Hz (${b.decisions} decisions in ~9 s)`);
  // Let it play on at 6x to see shots.
  await ev(() => window.game.setSpeed(6));
  await page.waitForTimeout(8000);
  b = await botInfo();
  console.log(`  at 6x: ${line(await summary())} | travelled ${b.travel.toFixed(0)} m, shots ${b.shots}, decisions ${b.decisions}, skipped ${b.skipped}`);
  check(b.shots >= 1, `the ONNX bot shoots (${b.shots} shots)`);
  // ACT v3: the policy's stick_up head reaches the sim through Sim.action_input.
  console.log(`  stick_up: mean P ${b.upProbMean?.toFixed(3)}, decisions with the stick up ${b.upDecisions} of ${b.decisions}, raised now ${b.stickUp}`);
  check(typeof b.upProbMean === "number" && b.upProbMean >= 0 && b.upProbMean <= 1, "the policy's stick_up_prob output is read every decision");
  console.log(`  brake: mean P ${b.brakeProbMean?.toFixed(3)}`);
  check(typeof b.brakeProbMean === "number" && b.brakeProbMean >= 0 && b.brakeProbMean <= 1, "the policy's brake_prob output is read every decision");
  // A raised bot stick drawn with the raised style: forced through the test hook (the policy decides it otherwise).
  await ev(() => { window.game.setSpeed(1); window.game.forceBotStickUp(true); });
  await page.waitForTimeout(400);
  const forced = await ev(() => ({ up: window.game.state.players[1].stick_up, inp: window.game.lastInputs[1]?.stick_up }));
  await page.screenshot({ path: path.join(out, "game-bot-stick-up.png") });
  console.log(`  forced raised bot stick: sim stick_up ${forced.up}, input ${forced.inp} (screenshot game-bot-stick-up.png)`);
  check(forced.up === true && forced.inp === true, "a raised bot stick reaches the sim (test hook)");
  await ev(() => window.game.forceBotStickUp(false));
  await page.keyboard.press("Escape");

  // ---- 4. clock end: vs the defender (it rarely attacks), human leads 1-0 (test hook), 40x ----
  await page.selectOption("#opponent", "defender");
  await page.click("#play");
  await page.waitForFunction(() => window.game.screen === "match", null, { timeout: 15000 });
  await ev(() => { window.game.setScore(1, 0); window.game.setSpeed(40); });
  console.log("\n[vs defender] score set to 1-0 by the test hook, loop at 40x");
  for (let i = 0; i < 4; i++) {
    await page.waitForTimeout(1500);
    console.log("  " + line(await summary()));
  }
  await page.waitForFunction(() => window.game.summary.phase === "finished", null, { timeout: 60000 });
  s = await summary();
  console.log(`  END ${line(s)}`);
  check(s.clock === "00:00" && !s.overtime && s.winner === 0, "match ends by the CLOCK with BLUE ahead");
  console.log("  game over:", (await page.textContent("#gameover .card")).replace(/\s+/g, " "));
  await page.screenshot({ path: path.join(out, "game-over-clock.png") });
  await page.keyboard.press("Escape");

  // ---- 5. fallback: model missing (404) and corrupt ----
  for (const [what, handler] of [
    ["missing", (route) => route.fulfill({ status: 404, body: "gone" })],
    ["corrupt", (route) => route.fulfill({ status: 200, contentType: "application/octet-stream", body: Buffer.from("this is not an onnx model") })],
    ["old-format (ACT v3 sidecar)", null],
  ]) {
    const p2 = await browser.newPage({ viewport: { width: 1280, height: 760 } });
    p2.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
    if (handler) await p2.route("**/bots/smoke.onnx", handler);
    else await p2.route("**/bots/smoke.json", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ obs_dim: 42, obs_version: 3, act_dim: 5, act_version: 3, outputs: ["move", "move_probs", "stick_dir", "dash_prob", "lift_prob"] }) }));
    await p2.goto(BASE);
    await p2.waitForSelector("body[data-ready='1']");
    await p2.selectOption("#opponent", "smoke");
    await p2.click("#play");
    await p2.waitForFunction(() => window.game.screen === "match" && window.game.summary.phase === "playing", null, { timeout: 30000 });
    await p2.waitForTimeout(2500); // the 3 s countdown is over: the fallback bot must be skating
    const bb = await p2.evaluate(() => window.game.bot);
    const ban = await p2.textContent("#banner");
    const vis = !(await p2.isHidden("#banner"));
    console.log(`\n[fallback, model ${what}] bot ${bb.label}; banner visible ${vis}: "${ban}"; bot travelled ${bb.travel.toFixed(1)} m`);
    check(bb.label === "scripted chaser" && vis && /Could not load/.test(ban) && bb.travel > 1, `model ${what}: falls back to the scripted chaser with a visible message`);
    if (what === "corrupt") await p2.screenshot({ path: path.join(out, "game-fallback.png") });
    await p2.close();
  }

  // ---- 6. the lab is still reachable from the menu ----
  await page.focus("#practice");
  await page.keyboard.press("Enter");
  await page.waitForURL(/lab\.html/);
  await page.waitForSelector("body[data-ready='1']", { timeout: 30000 });
  check(true, `Practice opens the feel lab (${page.url()})`);
  check(outside.length === 0, `no request escaped the subfolder or 404'd (${outside.length}${outside.length ? ": " + outside.slice(0, 3).join(", ") : ""})`);
} finally {
  await browser.close();
  server.close();
}
console.log(failed ? "\nVERIFY GAME FAILED" : "\nVERIFY GAME OK");
process.exit(failed ? 1 : 0);
