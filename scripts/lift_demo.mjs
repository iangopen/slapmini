// Lift in the sim (wasm build, the same code the browser runs): the puck at rest next to the blade,
// the stick swung through it, with and without lift held. Prints the puck's height over the hit.
//   wasm-pack build sim-wasm --release --target nodejs --out-dir pkg-node && node scripts/lift_demo.mjs
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { Sim } = require("../sim-wasm/pkg-node/sim_wasm.js");

function run(lift) {
  const sim = new Sim("", 1);
  const s = JSON.parse(sim.get_state());
  const pl = s.players[0];
  Object.assign(pl, { x: 0, y: 0, vx: 0, vy: 0, facing: 0, stick_angle: -1.2, stick_omega: 0 });
  // Puck at rest on the ice, 1.0 m out at angle 0 (inside the stick's reach), ahead of the cocked blade.
  Object.assign(s.puck, { x: 1.0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 });
  sim.set_state(JSON.stringify(s));
  let maxZ = 0, hitStep = -1, exit = 0;
  for (let k = 0; k < 120; k++) {
    // Swing counterclockwise through the puck (target well past it), lift held or not for the whole swing.
    sim.step(JSON.stringify([{ move_x: 0, move_y: 0, stick_target_angle: 1.2, dash: false, lift }]));
    const st = JSON.parse(sim.get_state());
    const sp = Math.hypot(st.puck.vx, st.puck.vy);
    if (hitStep < 0 && sp > 1) { hitStep = k; exit = sp; }
    maxZ = Math.max(maxZ, st.puck.z);
  }
  return { maxZ, hitStep, exit };
}
const a = run(true), b = run(false);
console.log(`lift held:     hit at step ${a.hitStep}, exit ${a.exit.toFixed(2)} m/s, max puck z ${a.maxZ.toFixed(3)} m`);
console.log(`lift not held: hit at step ${b.hitStep}, exit ${b.exit.toFixed(2)} m/s, max puck z ${b.maxZ.toFixed(3)} m`);
const ok = a.hitStep >= 0 && b.hitStep >= 0 && a.maxZ > 0.05 && b.maxZ === 0;
console.log(ok ? "LIFT OK: the puck leaves the ice only with lift held" : "LIFT CHECK FAILED");
process.exit(ok ? 0 : 1);
