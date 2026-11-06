// Every parameter the panel offers: change it, run a short scripted scenario that depends on it, print whether the
// behaviour changed, in the lab and in the running game match. Run: npm run verify-params [-- --report-only]
// The change goes through the panel's DOM input (#param-NAME) wherever a panel exists. A page without one (the game
// before w5b) can only edit its local copy window.game.params, which is the "before" column of the table.
import { freePort } from "./free_port.mjs";
import { chromium } from "playwright";
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.join(web, "verify-out");
mkdirSync(out, { recursive: true });
const reportOnly = process.argv.includes("--report-only");
const PORT = await freePort();
const server = spawn(`npx vite --port ${PORT} --strictPort`, { cwd: web, shell: true, stdio: ["ignore", "pipe", "pipe"], detached: process.platform !== "win32" });
await new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error("vite did not start")), 30000);
  server.stdout.on("data", (d) => { if (String(d).includes("Local:")) { clearTimeout(t); res(); } });
});
const FILE = JSON.parse(readFileSync(path.join(web, "..", "params.json"), "utf8"));

// name: [scenario, new value]. The scenarios live in the page (runScenario below).
const PLAN = {
  timestep: ["skate", 1 / 60], substeps: ["fastboard", 1], gravity: ["air", 5],
  rink_length: ["endboard", 60], rink_width: ["sideboard", 36], rink_corner_radius: ["corner", 4], board_restitution: ["sideboard", 0.8], board_friction: ["sideboard", 0.3],
  net_x: ["net", 22], net_width: ["netpost", 9], net_depth: ["netback", 1.2], net_height: ["nethigh", 0.5], net_restitution: ["netpost", 0.9],
  puck_radius: ["sideboard", 0.5], puck_mass: ["shot", 0.5], ice_friction_decel: ["glide", 2], ice_linear_drag: ["glide", 1], air_drag: ["air", 1], ice_restitution: ["air", 0.7],
  player_radius: ["body", 0.9], player_height: ["bodyhigh", 0.5], player_mass: ["body", 20], player_accel: ["skate", 8], player_max_speed: ["skate", 6], player_ice_drag: ["skate", 3],
  player_turn_rate: ["turn", 1], player_restitution: ["body", 0.9],
  stick_hand_x: ["shotnear", 0.6], stick_hand_y: ["shothand", 1.2], stick_kink_x: ["shotnear", 0.9], stick_kink_y: ["shotnear", -0.5], stick_heel_x: ["shot", 0.95], stick_heel_y: ["shot", -0.4], stick_toe_x: ["shot", 1.5], stick_toe_y: ["shot", -0.5], stick_shaft_radius: ["shotnear", 0.3], stick_heel_radius: ["shot", 0.3], stick_hit_rule: ["shot", 1],
  stick_inner: ["shotnear", 0.9], stick_length: ["shot", 0.7], stick_thickness: ["shot", 0.2], stick_mass: ["shot", 8], stick_max_ang_speed: ["flick", 10],
  stick_max_ang_accel: ["flick", 100], stick_restitution: ["shot", 0.9], stick_z_min: ["shot", 0.2], stick_z_max: ["shothigh", 0.2],
  max_puck_speed_air: ["shotlift", 8], max_puck_speed_ice_normal: ["shot", 5], max_puck_speed_ice_dash: ["shotdash", 5],
  dash_speed: ["dash", 20], dash_duration: ["shotdash2", 0.01], dash_cooldown: ["dash2", 0.1], lift_vz_factor: ["shotlift", 0.1], lift_max_vz: ["shotlift", 1], chip_max_speed: ["shotchip", 3],
  blue_line_x: ["display", 12], faceoff_x: ["display", 12], faceoff_y: ["display", 6], faceoff_circle_radius: ["display", 8], speed_display_scale: ["readout", 2],
  match_length_s: ["mclock", 120], mercy_lead: ["mmercy", 2], countdown_s: ["mcountdown", 1], goal_pause_s: ["mgoalpause", 1], player_spawn_x: ["mspawn", 8],
  player_coast_rate: ["skate", 3], dash_ramp_time: ["dash", 0.4], stick_max_ang_decel: ["flick", 300], stick_up_z_min: ["shotup", 0.9], stick_up_z_max: ["shotup", 0.5],
  // Added in w5b (absent from the file before it).
  player_turn_accel: ["turn", 60], player_brake_accel: ["reverse", 60], brake_speed: ["slow", 4], brake_decel: ["slow", 5],
};

// Runs inside the page. api = { state(), setState(s), step(inputs) }.
function runScenario(api, name) {
  const base = () => {
    const s = structuredClone(api.state());
    s.match_state = null; s.tick = 0; s.score = [0, 0];
    Object.assign(s.puck, { x: 0, y: 5, z: 0, vx: 0, vy: 0, vz: 0, in_net: null });
    const rest = { vx: 0, vy: 0, facing: 0, stick_angle: 0, stick_omega: 0, dash_timer: 0, dash_cooldown: 0, dash_prev: false, stick_up: false };
    Object.assign(s.players[0], rest, { x: -10, y: -8 });
    Object.assign(s.players[1], rest, { x: 10, y: -14 });
    return s;
  };
  const idle = { move_x: 0, move_y: 0, stick_target_angle: 0, dash: false, lift: false };
  const run = (s, steps, inp = () => idle) => {
    api.setState(s);
    const trace = [];
    for (let k = 0; k < steps; k++) {
      api.step([inp(k), idle]);
      if (k % 10 === 9) { const q = api.state(), a = q.players[0]; trace.push([q.puck.x, q.puck.y, q.puck.z, q.puck.vx, q.puck.vy, q.puck.vz, a.x, a.y, a.vx, a.vy, a.facing, a.stick_angle]); }
    }
    return trace;
  };
  const puck = (o) => { const s = base(); Object.assign(s.puck, o); return s; };
  // A swing into a resting puck: the stick starts 0.5 rad behind it at full speed.
  const shot = (o = {}, inp = {}, pl = {}) => {
    const s = base();
    Object.assign(s.puck, { x: -10 + 1.0, y: -8, ...o });
    Object.assign(s.players[0], { stick_angle: -0.5, stick_omega: 26, ...pl });
    return run(s, 60, (k) => ({ ...idle, stick_target_angle: 1.5, ...inp, dash: inp.dash ? k === 0 : false }));
  };
  const W = { ...idle, move_x: 1 };
  switch (name) {
    case "glide": return run(puck({ vx: 10 }), 120);
    case "air": return run(puck({ z: 1.2, vx: 8, vz: 5 }), 240);
    case "sideboard": return run(puck({ x: 0, y: 15, vx: 8, vy: 15 }), 120);
    case "fastboard": return run(puck({ x: 0, y: 19.5, vx: 3, vy: 40 }), 3).concat(run(puck({ x: -10, y: -8.6, vx: 60, vy: 0 }), 10));
    case "endboard": return run(puck({ x: 26, y: 8, vx: 20 }), 60);
    case "corner": return run(puck({ x: 22, y: 12, vx: 15, vy: 12 }), 120);
    case "net": return run(puck({ x: 15, y: 0, vx: 20 }), 120);
    case "netpost": return run(puck({ x: 26, y: 8, vy: -15 }), 90);
    case "netback": return run(puck({ x: 20, y: 0, vx: 25 }), 90);
    case "nethigh": return run(puck({ x: 20, y: 0, z: 0.75, vx: 22, vz: 1.2 }), 40);
    case "body": return run(puck({ x: -15, y: -8.2, vx: 12 }), 90);
    case "bodyhigh": return run(puck({ x: -8.5, y: -8.1, z: 0.8, vx: -12, vz: 1.5 }), 40);
    case "skate": return run(base(), 240, (k) => (k < 120 ? W : idle));
    case "turn": { const s = base(); Object.assign(s.players[0], { vx: 11, vy: 0 }); return run(s, 120, () => ({ ...idle, move_y: 1 })); }
    case "reverse": { const s = base(); Object.assign(s.players[0], { vx: 11, vy: 0 }); return run(s, 120, () => ({ ...idle, move_x: -1 })); }
    case "slow": { const s = base(); Object.assign(s.players[0], { vx: 11, vy: 0 }); return run(s, 120, () => ({ ...W, brake: true })); }
    case "dash": return run(base(), 60, (k) => ({ ...W, dash: k === 0 }));
    case "dash2": return run(base(), 120, (k) => ({ ...W, dash: k === 0 || k === 40 }));
    case "flick": return run(base(), 60, () => ({ ...idle, stick_target_angle: 2.5 }));
    case "shot": return shot();
    // The hand normally sits inside the body; moved out to 1.2 m it sweeps the puck while the blade stays short of it.
    case "shothand": return shot({ x: -10 + 0.85 }, { stick_target_angle: -0.9 }, { stick_angle: -2.4 });
    case "shotnear": return shot({ x: -10 + 0.85 });
    case "shothigh": return shot({ z: 0.22 });
    case "shotlift": return shot({}, { lift: true });
    case "shotchip": return shot({}, { lift: true, dash: true, move_x: 1 });
    case "shotdash": return shot({}, { dash: true, move_x: 1 });
    case "shotdash2": { // the dash window closes before the stick arrives: a normal shot instead of a dash shot
      const s = base();
      Object.assign(s.puck, { x: -9, y: -8 });
      Object.assign(s.players[0], { stick_angle: -2.0, stick_omega: 26 });
      return run(s, 60, (k) => ({ ...idle, stick_target_angle: 1.5, move_y: k < 2 ? 1 : 0, dash: k === 0 }));
    }
    case "shotup": return shot({ z: 0.7 }, { stick_up: true }, { stick_up: true });
    default: throw new Error("no scenario " + name);
  }
}

// In a running match: start it, edit the param while it runs, then read what the rules do (a live edit).
async function matchProbe(ev, newMatch, sc, apply) {
  await newMatch();
  const toPlaying = () => ev(() => {
    const a = window.__api; const s = structuredClone(a.state());
    s.match_state.phase = "playing"; s.match_state.phase_ticks = 0; a.setState(s);
  });
  switch (sc) {
    case "mclock": await toPlaying(); await apply(); return ev(() => window.__api.state().match_state.clock_ticks > 20000 ? "long" : "short");
    case "mcountdown": await apply(); return ev(() => window.__api.state().match_state.phase_ticks > 200 ? "long" : "short");
    case "mmercy":
      await toPlaying(); await apply();
      return ev(() => {
        const a = window.__api; const s = structuredClone(a.state());
        s.score = [1, 0]; Object.assign(s.puck, { x: 24.5, y: 0, z: 0, vx: 15, vy: 0, vz: 0, in_net: null }); a.setState(s);
        for (let i = 0; i < 20; i++) a.step([]);
        return a.state().match_state.phase;
      });
    default: // mgoalpause, mspawn
      await toPlaying(); await apply();
      return ev((c) => {
        const a = window.__api; const s = structuredClone(a.state());
        Object.assign(s.puck, { x: 24.5, y: 0, z: 0, vx: 15, vy: 0, vz: 0, in_net: null }); a.setState(s);
        for (let n = 0; n < 40 && a.state().match_state.phase === "playing"; n++) a.step([]);
        if (c === "mgoalpause") return a.state().match_state.phase_ticks;
        for (let i = 0; i < 2000 && a.state().match_state.phase !== "countdown"; i++) a.step([]);
        return a.state().players[0].x;
      }, sc);
  }
}

const launch = async () => { try { return await chromium.launch(); } catch (e) { const a = process.env.PW_CHROMIUM || "/opt/pw-browsers/chromium"; if (!existsSync(a)) throw e; return chromium.launch({ executablePath: a }); } };
const browser = await launch();
const results = {};
let failed = false;
const names = Object.keys(FILE.params);
for (const n of names) if (!PLAN[n]) { console.log(`FAIL: no audit plan for ${n}`); failed = true; }

async function audit(pageName) {
  const page = await browser.newPage({ viewport: { width: 1400, height: 760 } });
  page.on("pageerror", (e) => { console.log("PAGE ERROR", e.message); failed = true; });
  const ev = (fn, a) => page.evaluate(fn, a);
  const isLab = pageName === "lab";
  await page.goto(`http://localhost:${PORT}/${isLab ? "lab.html?mouse=cursor" : "index.html?camera=fit"}`);
  await page.waitForSelector("body[data-ready='1']");
  await ev((lab) => {
    window.__api = lab
      ? { state: () => window.lab.lab.state, setState: (s) => window.lab.lab.setState(s), step: (i) => { const L = window.lab.lab; const keep = L.inputs; if (i.length) L.inputs = i; L.step(0); L.inputs = keep; } }
      : { state: () => window.game.state, setState: (s) => window.game.setState(s), step: (i) => window.game.drillStep(i) };
  }, isLab);
  if (!isLab) {
    await page.click("#play");
    await page.waitForFunction(() => window.game.screen === "match");
    await ev(() => { window.__m0 = structuredClone(window.game.state.match_state); });
    if ((await page.$("#tune")) && (await page.locator("#tune").isHidden())) await page.keyboard.press("F2");
  }
  const hasPanel = (await page.$("#param-player_accel")) !== null;
  const setParam = async (n, v) => {
    if (hasPanel) await page.fill(`#param-${n}`, String(v));
    else await ev(([k, x]) => { window.game.params[k] = x; }, [n, v]);
  };
  const newMatch = () => isLab ? ev(() => window.lab.lab.startMatch()) : ev(() => {
    const s = structuredClone(window.game.state);
    s.match_state = { ...window.__m0 }; s.score = [0, 0]; window.game.setState(s);
  });
  const scenario = (sc) => ev(([f, s]) => { const fn = eval(`(${f})`); return JSON.stringify(fn(window.__api, s)); }, [runScenario.toString(), sc]);
  const pixels = () => ev(() => document.getElementById("rink").toDataURL("image/png"));
  const hold = () => ev(() => {
    const a = window.__api; const s = structuredClone(a.state());
    s.match_state = window.__m0 ? { ...window.__m0, phase: "finished", winner: 0 } : null; // nothing moves, no overlay in the canvas pixels we compare
    Object.assign(s.puck, { x: 0, y: 5, z: 0, vx: 0, vy: 0, vz: 0, in_net: null });
    for (const p of s.players) Object.assign(p, { vx: 0, vy: 0, stick_omega: 0 });
    a.setState(s);
  });
  const readout = () => ev(() => {
    const a = window.__api; const s = structuredClone(a.state()); s.puck.vx = 5; s.puck.vy = 0; s.puck.vz = 0; s.puck.z = 0;
    a.setState(s);
    return new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(document.getElementById("speed")?.textContent?.split(".")[0] ?? "no readout"))));
  });
  const row = {};
  for (const n of names) {
    if (!PLAN[n]) continue;
    const [sc, v] = PLAN[n];
    const orig = FILE.params[n].value;
    let before, after;
    if (sc.startsWith("m") && sc !== "") {
      before = await matchProbe(ev, newMatch, sc, async () => {});
      after = await matchProbe(ev, newMatch, sc, () => setParam(n, v));
      await setParam(n, orig);
    } else if (sc === "display") {
      await hold(); await page.waitForTimeout(80); before = await pixels();
      await setParam(n, v); await hold(); await page.waitForTimeout(80); after = await pixels();
      await setParam(n, orig);
    } else if (sc === "readout") {
      if (isLab) { before = await readout(); await setParam(n, v); after = await readout(); await setParam(n, orig); }
      else { before = "no speed readout on the game page"; after = before; }
    } else {
      before = await scenario(sc); await setParam(n, v); after = await scenario(sc); await setParam(n, orig);
    }
    const same = JSON.stringify(before) === JSON.stringify(after);
    const short = (x) => (typeof x === "string" && x.length > 40 ? "…" : x);
    row[n] = { sc, value: v, effect: !same, displayOnlyNA: sc === "readout" && !isLab, before: short(before), after: short(after) };
  }
  // Does a typed edit leave the keyboard usable? Edit a field in the panel, then press W without clicking anywhere.
  let keys = "n/a (no panel)", keysOk = false;
  if (hasPanel) {
    await ev(() => {
      const a = window.__api; const s = structuredClone(a.state());
      s.match_state = window.__m0 ? { ...window.__m0, phase: "playing", phase_ticks: 0 } : null;
      Object.assign(s.players[0], { x: -20, y: 0, vx: 0, vy: 0 });
      a.setState(s);
    });
    await page.fill("#param-player_accel", String(FILE.params.player_accel.value));
    await page.focus("#param-player_accel");
    await page.keyboard.down("KeyW");
    await page.waitForTimeout(350);
    const sp = await ev(() => { const p = window.__api.state().players[0]; return Math.hypot(p.vx, p.vy); });
    await page.keyboard.up("KeyW");
    keys = `speed ${sp.toFixed(2)} m/s after 0.35 s of W with the focus in the field`;
    keysOk = sp > 1;
  }
  await page.close();
  return { row, keys, keysOk, hasPanel };
}

try {
  for (const pageName of ["lab", "game"]) results[pageName] = await audit(pageName);
  const L = results.lab, G = results.game;
  console.log(`\nlab panel: ${L.hasPanel}; game panel: ${G.hasPanel}${G.hasPanel ? "" : " (the only thing to edit is the page's local copy window.game.params)"}`);
  console.log("| param | scenario | new value | lab | game |\n|---|---|---|---|---|");
  for (const n of names) {
    const l = L.row[n], g = G.row[n];
    if (!l) continue;
    const gs = g.displayOnlyNA ? "n/a (lab readout only)" : g.effect ? "yes" : "NO";
    console.log(`| ${n} | ${l.sc} | ${+l.value.toFixed(5)} | ${l.effect ? "yes" : "NO"} | ${gs} |`);
    // stick_inner only feeds the scripted bots (hit radius) since the capsule chain took over collision.
    if (!reportOnly && n !== "stick_inner" && !(l.effect && (g.effect || g.displayOnlyNA))) failed = true;
  }
  const cnt = (r) => names.filter((n) => r.row[n]?.effect).length;
  console.log(`\neffect: lab ${cnt(L)}/${names.length}, game ${cnt(G)}/${names.length}`);
  console.log(`keyboard after editing a field: lab: ${L.keys}; game: ${G.keys}`);
  if (!reportOnly && !(L.keysOk && G.keysOk)) failed = true;
  writeFileSync(path.join(out, "param-audit.json"), JSON.stringify(results, null, 1));
} finally {
  await browser.close();
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"]);
  else { try { process.kill(-server.pid); } catch { /* gone */ } }
}
console.log(failed ? "VERIFY PARAMS FAILED" : reportOnly ? "report only" : "VERIFY PARAMS OK");
process.exit(failed ? 1 : 0);
