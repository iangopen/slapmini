// Player/puck scale in the game: drawn radii (px) at the current camera, plus a screenshot with the puck
// next to the player. Run: node scripts/scale_shot.mjs OUT.png [query]   (own Vite dev server)
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outFile = path.resolve(web, process.argv[2] ?? "verify-out/w4-scale.png");
const query = process.argv[3] ?? "";
mkdirSync(path.dirname(outFile), { recursive: true });
const PORT = 5190;
const server = spawn(`npx vite --port ${PORT} --strictPort`, { cwd: web, shell: true, stdio: ["ignore", "pipe", "pipe"], detached: process.platform !== "win32" });
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite did not start")), 30000);
  server.stdout.on("data", (d) => { if (String(d).includes("Local:")) { clearTimeout(t); res(); } });
});
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 760 } });
  page.on("pageerror", (e) => console.log("PAGE ERROR", e.message));
  await page.goto(`http://localhost:${PORT}/${query}`);
  await page.waitForSelector("body[data-ready='1']");
  await page.selectOption("#opponent", "defender");
  await page.click("#play");
  await page.waitForFunction(() => window.game.screen === "match" && window.game.summary.phase === "playing", null, { timeout: 15000 });
  // Freeze a picture: the puck resting 1.0 m in front of BLUE (as at the forehand in the clips).
  const info = await page.evaluate(() => {
    const g = window.game;
    g.setSpeed(1);
    const s = structuredClone(g.state);
    const p = s.players[0];
    Object.assign(p, { x: -6, y: 0, vx: 0, vy: 0 });
    Object.assign(s.puck, { x: -5.0, y: 0.0, vx: 0, vy: 0, z: 0 });
    g.setState(s);
    const P = g.params, v = g.view;
    return { s: v.s, rot: v.rot, player_radius: P.player_radius, puck_radius: P.puck_radius, drawn: g.w4?.radii ?? null };
  });
  await page.waitForTimeout(50);
  await page.screenshot({ path: outFile });
  const pr = info.drawn ?? { player: info.player_radius * info.s, puck: Math.max(info.puck_radius * info.s, 6) };
  console.log(`scale ${info.s.toFixed(3)} px/m${query ? " (" + query + ")" : ""}`);
  console.log(`sim radii: player ${info.player_radius} m, puck ${info.puck_radius} m (ratio ${(info.player_radius / info.puck_radius).toFixed(3)})`);
  console.log(`true radii at this scale: player ${(info.player_radius * info.s).toFixed(2)} px, puck ${(info.puck_radius * info.s).toFixed(2)} px`);
  console.log(`drawn radii: player ${pr.player.toFixed(2)} px, puck ${pr.puck.toFixed(2)} px (drawn ratio ${(pr.player / pr.puck).toFixed(3)})`);
  console.log(`screenshot ${outFile}`);
} finally {
  await browser.close();
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], { stdio: "ignore" });
  else { try { process.kill(-server.pid, "SIGTERM"); } catch { server.kill(); } }
}
