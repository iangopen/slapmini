// Native-vs-wasm observation check: the browser must build the exact training observation.
//   cargo run --release -p sim-wasm --example obs_fixture > target/obs_fixture.jsonl
//   wasm-pack build sim-wasm --release --target nodejs --out-dir pkg-node
//   node scripts/obs_crosscheck.mjs target/obs_fixture.jsonl
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { Sim } = require("../sim-wasm/pkg-node/sim_wasm.js");

const path = process.argv[2] ?? "target/obs_fixture.jsonl";
const lines = readFileSync(path, "utf8").split("\n").filter((l) => l.trim());
const sim = new Sim("", 2);
let fields = 0, mismatches = 0, matchStates = 0, actions = 0, actionMismatches = 0, brakes = 0;
const up = [0, 0, 0, 0]; // stick_up cases: none, own only, opponent only, both (OBS v3 fields 40, 41)
if (Sim.obs_dim() !== 42 || Sim.obs_version() !== 3) throw new Error(`wasm obs v${Sim.obs_version()}/${Sim.obs_dim()}, expected v3/42`);
for (const line of lines) {
  const rec = JSON.parse(line);
  sim.set_state(rec.state); // the native State JSON text, unchanged
  if (JSON.parse(rec.state).match_state) matchStates++;
  const got = new Uint32Array(sim.observe(rec.player, new Float32Array(rec.style)).buffer);
  const st = JSON.parse(rec.state), me = st.players[rec.player], op = st.players.find((p) => p.team !== me.team);
  up[(me.stick_up ? 1 : 0) + (op?.stick_up ? 2 : 0)]++;
  const ai = sim.action_input(rec.player, ...rec.act);
  actions++;
  if (rec.act[5]) brakes++;
  if (ai !== rec.input) { actionMismatches++; if (actionMismatches <= 5) console.log(`MISMATCH action: wasm ${ai} native ${rec.input}`); }
  if (got.length !== rec.bits.length) throw new Error(`length ${got.length} vs ${rec.bits.length}`);
  for (let k = 0; k < got.length; k++) {
    fields++;
    if (got[k] !== rec.bits[k]) {
      mismatches++;
      if (mismatches <= 5) console.log(`MISMATCH obs ${lines.indexOf(line)} field ${k}: wasm ${got[k].toString(16)} native ${rec.bits[k].toString(16)}`);
    }
  }
}
console.log(`obs crosscheck: ${lines.length} observations (${matchStates} in match mode; stick up: none ${up[0]}, own only ${up[1]}, opponent only ${up[2]}, both ${up[3]}), ${fields} fields, ${mismatches} bit mismatches -> ${mismatches ? "FAIL" : "IDENTICAL"}`);
console.log(`action crosscheck: ${actions} actions (${brakes} with the brake), ${actionMismatches} mismatches`);
process.exit(mismatches || actionMismatches || brakes === 0 || lines.length === 0 || up.some((c) => c === 0) ? 1 : 0);
