// Native-vs-wasm determinism check. Build first:
//   wasm-pack build sim-wasm --release --target nodejs --out-dir pkg-node
// then: node scripts/wasm_crosscheck.mjs   and compare with
//   cargo test -p sim --release crosscheck_hash_native -- --nocapture
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { Sim } = require("../sim-wasm/pkg-node/sim_wasm.js");

const sim = new Sim("", 2);
for (let t = 0; t < 100000; t++) {
  const inputs = [0, 1].map((k) => ({
    move_x: (Math.floor(t / 60 + k) % 3) - 1,
    move_y: (Math.floor(t / 90 + k) % 3) - 1,
    stick_target_angle: ((t + k * 100) % 628) * 0.01 - 3.14,
    dash: t % 240 === k * 7,
    lift: t % 500 < 20,
  }));
  sim.step(JSON.stringify(inputs));
}
console.log(`wasm crosscheck hash after 100000 steps: ${sim.state_hash()}`);
