// w4: attribute the crosscheck hash changes. Runs the three existing crosscheck runs with the NEW code (stick_up
// present but never raised) and params.json, with player_radius set back to its pre-w4 value. If they reproduce the
// pre-w4 hashes, the radius change alone moved them.  Run: node scripts/radius_attribution.mjs [old_radius]
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { Sim } = require("../sim-wasm/pkg-node/sim_wasm.js");

const oldR = Number(process.argv[2] ?? 0.495466);
const pf = JSON.parse(readFileSync(new URL("../params.json", import.meta.url), "utf8"));
const newR = pf.params.player_radius.value;
pf.params.player_radius.value = oldR;
const P = JSON.stringify(pf);

const inputsAt = (t) => [0, 1].map((k) => ({
  move_x: (Math.floor(t / 60 + k) % 3) - 1, move_y: (Math.floor(t / 90 + k) % 3) - 1,
  stick_target_angle: ((t + k * 100) % 628) * 0.01 - 3.14, dash: t % 240 === k * 7, lift: t % 500 < 20,
}));
const free = new Sim(P, 2);
for (let t = 0; t < 100000; t++) free.step(JSON.stringify(inputsAt(t)));
const m = new Sim(P, 2);
m.start_match(2);
for (let t = 0; t < 100000; t++) m.step(JSON.stringify(inputsAt(t)));
const s10 = new Sim(P, 2);
for (let t = 0; t < 100000; t++) {
  const st = JSON.parse(s10.get_state());
  s10.step(JSON.stringify(st.players.map((pl, k) => {
    const dx = st.puck.x - pl.x, dy = st.puck.y - pl.y, d = Math.max(Math.sqrt(dx * dx + dy * dy), 1e-9);
    const phase = (t + k * 97) % 600, coasting = phase >= 480, sense = (Math.floor(t / 300) + k) % 2 === 0 ? 1.0 : -1.0;
    return { move_x: coasting ? 0.0 : dx / d, move_y: coasting ? 0.0 : dy / d, stick_target_angle: d < 2.0 ? pl.stick_angle + sense * 2.5 : pl.stick_angle,
      dash: phase % 150 < 3 && !coasting, lift: t % 700 < 15 };
  })));
}
const want = { free: "ed2b9c3b2c52da91", match: "ef5c44a7fc4de600", s10: "b19f6906467b50c1" };
const got = { free: free.state_hash(), match: m.state_hash(), s10: s10.state_hash() };
console.log(`new code, player_radius ${oldR} (pre-w4; params.json now ${newR}):`);
for (const k of Object.keys(want)) console.log(`  ${k.padEnd(5)} ${got[k]}  pre-w4 ${want[k]}  ${got[k] === want[k] ? "IDENTICAL" : "DIFFERENT"}`);
process.exit(Object.keys(want).every((k) => got[k] === want[k]) ? 0 : 1);
