// Space (lift) diagnosis: press Space in the game and in the lab through the real UI and print what
// the sim receives each step. Run: node scripts/diag_space.mjs (starts its own Vite dev server).
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PORT = 5181;
const server = spawn(`npx vite --port ${PORT} --strictPort`, { cwd: web, shell: true, stdio: ["ignore", "pipe", "pipe"], detached: process.platform !== "win32" });
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite did not start")), 30000);
  server.stdout.on("data", (d) => { if (String(d).includes("Local:")) { clearTimeout(t); res(); } });
});
const launch = async () => { try { return await chromium.launch(); } catch (e) { const a = process.env.PW_CHROMIUM || "/opt/pw-browsers/chromium"; if (!existsSync(a)) throw e; return chromium.launch({ executablePath: a }); } };
const browser = await launch();
let liftSeen = { game: false, lab: false };
const labCases = [];
try {
  // ---- game ----
  const page = await browser.newPage({ viewport: { width: 1280, height: 760 } });
  page.on("pageerror", (e) => console.log("PAGE ERROR", e.message));
  await page.goto(`http://localhost:${PORT}/`);
  await page.waitForSelector("body[data-ready='1']");
  await page.selectOption("#opponent", "idle").catch(() => page.selectOption("#opponent", "defender"));
  await page.click("#play"); // the mouse, as a player would
  await page.waitForFunction(() => window.game.screen === "match" && window.game.summary.phase === "playing", null, { timeout: 15000 });
  const g = () => page.evaluate(() => ({ active: window.game.activeElement, started: window.game.matchesStarted, phase: window.game.summary.phase, tick: window.game.summary.tick,
    human: window.game.lastInputs[0], scrollY: window.scrollY }));
  console.log("[game] after clicking Play:", JSON.stringify(await g()));
  await page.keyboard.down("Space");
  for (let k = 0; k < 6; k++) {
    await page.waitForTimeout(50);
    const s = await g();
    liftSeen.game ||= !!s.human?.lift;
    console.log(`[game] Space held, sample ${k}: tick ${s.tick} lift ${s.human?.lift} phase ${s.phase} matchesStarted ${s.started} active ${s.active}`);
  }
  await page.keyboard.up("Space");
  await page.waitForTimeout(300);
  console.log("[game] after Space released:", JSON.stringify(await g()));

  // ---- game, match started from the keyboard (Enter on the focused Play button) ----
  await page.keyboard.press("Escape");
  await page.waitForFunction(() => window.game.screen === "menu");
  await page.focus("#play");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => window.game.screen === "match" && window.game.summary.phase === "playing", null, { timeout: 15000 });
  const before2 = await g();
  console.log("[game/keyboard start] after Enter on Play:", JSON.stringify(before2));
  await page.keyboard.down("Space");
  await page.waitForTimeout(150);
  const held2 = await g();
  await page.keyboard.up("Space");
  await page.waitForTimeout(600);
  const after2 = await g();
  console.log(`[game/keyboard start] Space held: lift ${held2.human?.lift}, active ${held2.active}; after release: matchesStarted ${before2.started} -> ${after2.started}, phase ${after2.phase}, tick ${after2.tick}`);

  // ---- lab ----
  const lab = await browser.newPage({ viewport: { width: 1280, height: 760 } });
  lab.on("pageerror", (e) => console.log("PAGE ERROR", e.message));
  await lab.goto(`http://localhost:${PORT}/lab.html`);
  await lab.waitForSelector("body[data-ready='1']");
  for (const [what, prep] of [
    ["after clicking the rink canvas", async () => lab.mouse.click(500, 380)],
    // (since w2 the canvas click captures the mouse: Esc releases it so the panel is clickable)
    ["after clicking the Match mode checkbox", async () => { await lab.keyboard.press("Escape"); await lab.click("#match-mode"); }],
    ["after clicking the Record button and stopping", async () => { await lab.click("#rec"); await lab.click("#rec"); }],
  ]) {
    await prep();
    const before = await lab.evaluate(() => ({ active: document.activeElement?.tagName + "#" + document.activeElement?.id, match: document.getElementById("match-mode").checked, mode: window.lab.lab.mode }));
    await lab.keyboard.down("Space");
    await lab.waitForTimeout(150);
    const held = await lab.evaluate(() => ({ lift: window.lab.lab.inputs[0].lift, tick: window.lab.lab.state.tick }));
    await lab.keyboard.up("Space");
    await lab.waitForTimeout(100);
    const after = await lab.evaluate(() => ({ match: document.getElementById("match-mode").checked, mode: window.lab.lab.mode, scrollY: window.scrollY }));
    liftSeen.lab ||= held.lift;
    labCases.push(held.lift && before.match === after.match && before.mode === after.mode);
    console.log(`[lab] ${what}: focus ${before.active}; while Space held lift=${held.lift} (tick ${held.tick}); match checkbox ${before.match} -> ${after.match}; mode ${before.mode} -> ${after.mode}`);
  }
} finally {
  await browser.close();
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], { stdio: "ignore" });
  else { try { process.kill(-server.pid, "SIGTERM"); } catch { server.kill(); } }
}
console.log(`\nlift reached the sim: game ${liftSeen.game}, lab ${liftSeen.lab}`);
const ok = liftSeen.game && labCases.length === 3 && labCases.every(Boolean);
console.log(ok ? "SPACE OK: lift reaches the sim in every case and no focused control reacts to Space" : `SPACE NOT OK (lab cases ${JSON.stringify(labCases)})`);
process.exit(ok ? 0 : 1);
