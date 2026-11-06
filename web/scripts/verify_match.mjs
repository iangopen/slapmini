// Headless match-mode verification: three full 1v1 matches in the feel lab with the loop sped up
// (mercy end, clock end with a winner, tied at 00:00 → overtime → next goal), plus practice tools
// off in match mode, inputs ignored in the countdown, Enter = rematch, and record/replay across a
// goal reset. Screenshots in web/verify-out/match-*.png; the recorded match scenario + CSV too.
// Run: npm run wasm && npm run verify-match
import { freePort } from "./free_port.mjs";
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(here, "..");
const out = path.join(webDir, "verify-out");
mkdirSync(out, { recursive: true });
const PORT = await freePort();
const URL_ = `http://localhost:${PORT}/lab.html?mode=match`;

const server = spawn(`npx vite --port ${PORT} --strictPort`, {
  cwd: webDir,
  shell: true,
  stdio: ["ignore", "pipe", "pipe"],
  detached: process.platform !== "win32",
});
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite did not start")), 30000);
  server.stdout.on("data", (d) => {
    if (String(d).includes("Local:")) { clearTimeout(t); res(); }
  });
});

const sha = (s) => createHash("sha256").update(s).digest("hex");
let failed = false;
const check = (cond, msg) => { console.log(`${cond ? "PASS" : "FAIL"}: ${msg}`); if (!cond) failed = true; };

async function launch() {
  try {
    return await chromium.launch();
  } catch (e) {
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
  await page.goto(URL_);
  await page.waitForSelector("body[data-ready='1']");
  const ev = (fn, arg) => page.evaluate(fn, arg);
  const setSpeed = (n) => ev((n) => window.lab.setSpeed(n), n);

  const status = () => ev(() => {
    const l = window.lab.lab;
    const m = l.state.match_state;
    return { score: [...l.state.score], clock: m ? window.lab.formatClock(m.clock_ticks, l.p.timestep) : null, phase: m?.phase ?? "free", overtime: m?.overtime ?? false, winner: m?.winner ?? null, tick: l.state.tick };
  });
  const show = async (label) => {
    const s = await status();
    console.log(`  ${label.padEnd(34)} score ${s.score[0]}-${s.score[1]}  clock ${s.clock}  phase ${s.phase.padEnd(10)}${s.overtime ? " OVERTIME" : ""}${s.winner !== null ? `  winner team ${s.winner}` : ""}`);
    return s;
  };
  const waitPhase = (ph, timeout = 60000) => page.waitForFunction((ph) => window.lab.lab.state.match_state?.phase === ph, ph, { timeout, polling: 16 });
  /** In the playing phase, slide the puck into the net `team` attacks (team 0 → +x) and wait for the goal. */
  const shoot = async (team) => {
    await waitPhase("playing");
    const before = await ev((t) => window.lab.lab.state.score[t], team);
    await ev((team) => {
      const l = window.lab.lab;
      const s = structuredClone(l.state);
      const sign = team === 0 ? 1 : -1;
      s.puck = { x: sign * (l.p.net_x - 1), y: 0, z: 0, vx: sign * 8, vy: 0, vz: 0, in_net: null };
      l.setState(s);
    }, team);
    await page.waitForFunction(([t, b]) => window.lab.lab.state.score[t] > b, [team, before], { timeout: 30000, polling: 16 });
  };

  // ---------- match 1: mercy ----------
  console.log("\n[match 1: mercy rule, a 5-goal lead]");
  await setSpeed(1);
  let s = await show("start (?mode=match)");
  check(s.phase === "countdown" && s.clock === "05:00" && s.score.join() === "0,0", "match starts in the countdown at 05:00, 0-0");
  check(await ev(() => !window.lab.lab.toolsEnabled), "practice tools are disabled in match mode");
  // Inputs ignored during the countdown (hold D), practice keys do nothing (F/Q/R).
  const p0 = await ev(() => ({ ...window.lab.lab.state.players[0] }));
  const pk0 = await ev(() => ({ ...window.lab.lab.state.puck }));
  const plusX = (await ev(() => window.lab.view.rot)) ? "KeyW" : "KeyD"; // the key that moves toward +x in this view
  await page.keyboard.down(plusX);
  await page.mouse.move(900, 200);
  await page.keyboard.press("KeyF");
  await page.keyboard.press("KeyQ");
  await page.keyboard.press("KeyR");
  await page.waitForTimeout(600);
  const p1 = await ev(() => ({ ...window.lab.lab.state.players[0], phase: window.lab.lab.state.match_state.phase }));
  const pk1 = await ev(() => ({ ...window.lab.lab.state.puck }));
  await page.screenshot({ path: path.join(out, "match-1-countdown.png") });
  check(p1.phase === "countdown" && p1.x === p0.x && p1.y === p0.y && p1.stick_angle === p0.stick_angle, `player frozen in the countdown while D held and the mouse moved (x ${p1.x}, stick ${p1.stick_angle.toFixed(3)})`);
  check(pk1.x === pk0.x && pk1.y === pk0.y && pk1.z === pk0.z, "F/Q/R did not move the puck in match mode");
  await waitPhase("playing");
  await page.waitForTimeout(300);
  const p2 = await ev(() => window.lab.lab.state.players[0]);
  check(p2.x !== p0.x, `inputs act once playing (player x ${p0.x} → ${p2.x.toFixed(2)})`);
  await page.keyboard.up(plusX);
  await setSpeed(20);
  for (let g = 1; g <= 5; g++) {
    await shoot(0);
    s = await show(`goal ${g} (team 0)`);
    if (g === 2) {
      await waitPhase("playing");
      await setSpeed(1);
      await page.waitForTimeout(500);
      await show("mid-match (screenshot)");
      await page.screenshot({ path: path.join(out, "match-2-scoreboard-mid.png") });
      await setSpeed(20);
    }
    if (g === 3) {
      await setSpeed(1);
      await page.screenshot({ path: path.join(out, "match-3-goal-pause.png") });
      await setSpeed(20);
    }
    if (g < 5) check(s.phase === "goal_pause", `after goal ${g}: goal pause, no winner yet`);
  }
  check(s.phase === "finished" && s.winner === 0 && s.score.join() === "5,0" && s.clock !== "00:00", "5-0 ends the match by the mercy rule before the clock runs out");
  await page.waitForTimeout(200);
  await page.screenshot({ path: path.join(out, "match-4-game-over-mercy.png") });
  const fin = await status();
  await page.waitForTimeout(500);
  check((await status()).score.join() === fin.score.join() && (await status()).phase === "finished", "finished match stays finished");

  // ---------- match 2: clock ----------
  console.log("\n[match 2: clock runs out with a winner; Enter = rematch; record/replay across a goal reset]");
  await page.keyboard.press("Enter");
  s = await show("after Enter");
  check(s.phase === "countdown" && s.clock === "05:00" && s.score.join() === "0,0", "Enter starts a rematch");
  await setSpeed(20);
  await shoot(1);
  await show("goal (team 1)");
  // Record from the goal pause through the reset + countdown into play, with real key/mouse input.
  await setSpeed(2);
  await page.click("#rec");
  await page.mouse.move(700, 300);
  await page.keyboard.down("KeyD");
  await page.keyboard.down("KeyW");
  await waitPhase("playing", 20000);
  await page.waitForTimeout(500);
  await page.keyboard.down("Space");
  await page.mouse.move(800, 500);
  await page.waitForTimeout(400);
  await page.keyboard.up("Space");
  await page.keyboard.up("KeyW");
  await page.keyboard.up("KeyD");
  await page.click("#rec");
  const rec = await ev(() => {
    const r = window.lab.lab.lastRecording;
    return { csv: window.lab.toCsv(r.samples), sc: JSON.stringify(r.scenario, null, 2) + "\n", endHash: r.endHash };
  });
  const sc = JSON.parse(rec.sc);
  writeFileSync(path.join(out, "match_run.csv"), rec.csv);
  writeFileSync(path.join(out, "match_run.scenario.json"), rec.sc);
  await ev(() => (window.lab.lab.lastReplay = null));
  await page.click("#replay");
  await page.waitForFunction(() => window.lab.lab.mode === "play" && window.lab.lab.lastReplay, null, { timeout: 60000 });
  const rep = await ev(() => ({ csv: window.lab.toCsv(window.lab.lab.lastReplay.samples), endHash: window.lab.lab.lastReplay.endHash }));
  writeFileSync(path.join(out, "match_run.replay.csv"), rep.csv);
  console.log(`  recorded ${sc.steps} steps from phase '${sc.initial_state.match_state.phase}', ${sc.inputs.length} input events`);
  console.log(`  recorded csv sha256 ${sha(rec.csv)}, end state hash ${rec.endHash}`);
  console.log(`  replayed csv sha256 ${sha(rep.csv)}, end state hash ${rep.endHash}`);
  check(sc.initial_state.match_state.phase === "goal_pause", "recording starts in the goal pause (scenario carries match_state)");
  check(sha(rec.csv) === sha(rep.csv) && rec.endHash === rep.endHash, "match recording replays byte-identically across the goal reset");
  await setSpeed(100);
  await page.waitForFunction(() => window.lab.lab.state.match_state.clock_ticks < 18000, null, { timeout: 60000, polling: 16 });
  await show("half time");
  await waitPhase("finished", 120000);
  s = await show("clock end");
  check(s.phase === "finished" && s.winner === 1 && s.clock === "00:00" && !s.overtime && s.score.join() === "0,1", "clock reaches 00:00 with team 1 ahead: team 1 wins");
  await page.screenshot({ path: path.join(out, "match-5-game-over-clock.png") });

  // ---------- match 3: overtime ----------
  console.log("\n[match 3: tied at 00:00 → overtime → next goal wins]");
  await page.keyboard.press("Enter");
  await show("after Enter");
  await setSpeed(20);
  await shoot(0);
  await show("goal (team 0)");
  await shoot(1);
  await show("goal (team 1)");
  await waitPhase("playing");
  await setSpeed(100);
  await page.waitForFunction(() => window.lab.lab.state.match_state.overtime, null, { timeout: 120000, polling: 16 });
  await setSpeed(1);
  s = await show("regulation over, tied");
  check(s.phase === "playing" && s.overtime && s.clock === "00:00" && s.winner === null, "tied at 00:00: overtime, play continues");
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(out, "match-6-overtime.png") });
  s = await show("overtime +0.8 s");
  check(s.phase === "playing" && s.clock === "00:00", "overtime has no clock");
  await setSpeed(20);
  await shoot(1);
  s = await show("overtime goal (team 1)");
  check(s.phase === "finished" && s.winner === 1 && s.overtime && s.score.join() === "1,2", "the next goal decides it: team 1 wins in overtime");
  await page.screenshot({ path: path.join(out, "match-7-game-over-overtime.png") });

  // ---------- back to free play ----------
  await page.uncheck("#match-mode");
  s = await show("match mode off");
  check(s.phase === "free" && (await ev(() => window.lab.lab.toolsEnabled)), "unchecking match mode returns to free play with the practice tools on");
} finally {
  await browser?.close();
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], { stdio: "ignore" });
  else {
    try { process.kill(-server.pid, "SIGTERM"); } catch { server.kill(); }
  }
}
console.log(failed ? "\nVERIFY MATCH FAILED" : "\nVERIFY MATCH OK");
process.exit(failed ? 1 : 0);
