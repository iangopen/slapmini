// Native-vs-wasm determinism check. Build first:
//   wasm-pack build sim-wasm --release --target nodejs --out-dir pkg-node
// then: node scripts/wasm_crosscheck.mjs   and compare with
//   cargo test -p sim --release crosscheck_hash_native -- --nocapture
//   cargo test -p sim --release --test s10_stress s10_determinism -- --nocapture   (the s10 run below)
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { Sim } = require("../sim-wasm/pkg-node/sim_wasm.js");

const inputsAt = (t) => [0, 1].map((k) => ({
  move_x: (Math.floor(t / 60 + k) % 3) - 1,
  move_y: (Math.floor(t / 90 + k) % 3) - 1,
  stick_target_angle: ((t + k * 100) % 628) * 0.01 - 3.14,
  dash: t % 240 === k * 7,
  lift: t % 500 < 20,
}));

const sim = new Sim("", 2);
for (let t = 0; t < 100000; t++) sim.step(JSON.stringify(inputsAt(t)));
console.log(`wasm crosscheck hash after 100000 steps: ${sim.state_hash()}`);

// Match rules on (compare with: cargo test -p sim --release crosscheck_hash_native_match -- --nocapture).
const m = new Sim("", 2);
m.start_match(2);
for (let t = 0; t < 100000; t++) m.step(JSON.stringify(inputsAt(t)));
const ms = JSON.parse(m.get_state()).match_state;
console.log(`wasm match crosscheck hash after 100000 steps: ${m.state_hash()} (${JSON.stringify(ms)})`);

// s10 models (skating, coasting, dashing, stick spins and shots): same inputs as sim/tests/s10_stress.rs s10_inputs,
// computed from the state every step with + - * / sqrt only (bit-identical in JS and Rust).
const s10 = new Sim("", 2);
for (let t = 0; t < 100000; t++) {
  const st = JSON.parse(s10.get_state());
  const inp = st.players.map((pl, k) => {
    const dx = st.puck.x - pl.x, dy = st.puck.y - pl.y;
    const d = Math.max(Math.sqrt(dx * dx + dy * dy), 1e-9);
    const phase = (t + k * 97) % 600;
    const coasting = phase >= 480;
    const sense = (Math.floor(t / 300) + k) % 2 === 0 ? 1.0 : -1.0;
    return {
      move_x: coasting ? 0.0 : dx / d,
      move_y: coasting ? 0.0 : dy / d,
      stick_target_angle: d < 2.0 ? pl.stick_angle + sense * 2.5 : pl.stick_angle,
      dash: phase % 150 < 3 && !coasting,
      lift: t % 700 < 15,
    };
  });
  s10.step(JSON.stringify(inp));
}
console.log(`wasm s10 crosscheck hash after 100000 steps: ${s10.state_hash()} (score ${JSON.stringify(JSON.parse(s10.get_state()).score)})`);

// w4 raised stick: the s10 inputs plus stick_up toggling in 40-step blocks (sim/tests/stick_up.rs stick_up_inputs;
// u32 hash x * 2654435761 >> 16 via Math.imul, identical to Rust's wrapping_mul).
// Compare with: cargo test -p sim --release --test stick_up stick_up_determinism -- --nocapture
const su = new Sim("", 2);
for (let t = 0; t < 100000; t++) {
  const st = JSON.parse(su.get_state());
  const inp = st.players.map((pl, k) => {
    const dx = st.puck.x - pl.x, dy = st.puck.y - pl.y;
    const d = Math.max(Math.sqrt(dx * dx + dy * dy), 1e-9);
    const phase = (t + k * 97) % 600;
    const coasting = phase >= 480;
    const sense = (Math.floor(t / 300) + k) % 2 === 0 ? 1.0 : -1.0;
    const x = Math.floor(t / 40) * 2 + k;
    const up = ((Math.imul(x, 2654435761) >>> 0) >>> 16) % 3 === 0;
    return {
      move_x: coasting ? 0.0 : dx / d,
      move_y: coasting ? 0.0 : dy / d,
      stick_target_angle: d < 2.0 ? pl.stick_angle + sense * 2.5 : pl.stick_angle,
      dash: phase % 150 < 3 && !coasting,
      lift: t % 700 < 15,
      stick_up: up,
    };
  });
  su.step(JSON.stringify(inp));
}
console.log(`wasm stick_up crosscheck hash after 100000 steps: ${su.state_hash()} (score ${JSON.stringify(JSON.parse(su.get_state()).score)})`);
