// Brake (Space), built in the web layer from the sim's own controls: the sim has no brake input, and
// adding one would change the physics before the first training run.
//
// While the brake is held the player sends a move input OPPOSITE to its current velocity. The sim applies
// a move input of length m as a velocity change of m · player_accel · timestep per step (and no coasting
// while a move key is held), so m = min(1, speed / (player_accel · timestep)) stops the player exactly on
// the step it would otherwise overshoot: full strength while fast, tapering to zero, never reversing.
// Below 0.05 m/s no move is sent (the coast takes over). The stopping time is therefore the sim's skating
// acceleration run backwards: a PLACEHOLDER, the real game's braking is unmeasured (CLAUDE.md lists
// "W then S at top speed" as a missing recording). Bots do not brake.

export const BRAKE_MIN_SPEED = 0.05; // m/s

/** World move input that brakes a player moving at (vx, vy). */
export function brakeMove(vx: number, vy: number, accel: number, timestep: number): [number, number] {
  const s = Math.hypot(vx, vy);
  if (!(s >= BRAKE_MIN_SPEED) || !(accel > 0) || !(timestep > 0)) return [0, 0];
  const m = Math.min(1, s / (accel * timestep));
  return [(-vx / s) * m, (-vy / s) * m];
}
