// Skill → handicaps for browser bots. ONE place for the mapping.
//
// UNCALIBRATED: these numbers are guesses. The plan is to calibrate them with the rating ladder
// (python -m train.league): rate a policy at several skill settings and pick values that space the
// ratings evenly. Until then, "skill 50" means nothing more than "half of the handicaps below".
//
// Skill 100 = no handicap at all (the bot plays exactly as trained / scripted).

export interface Handicap {
  /** Sim steps (120 Hz) between what the bot sees and the true state (perception delay). */
  delaySteps: number;
  /** Standard deviation of Gaussian noise added to the stick target angle, rad (redrawn each decision). */
  stickNoise: number;
  /** Mistakes per second of play: the bot skates in a random direction and holds its stick still. */
  mistakeRate: number;
  /** Length of one mistake, sim steps. */
  mistakeSteps: number;
}

export const SKILL_MIN = 0;
export const SKILL_MAX = 100;

export function handicapFor(skill: number): Handicap {
  const h = 1 - Math.min(SKILL_MAX, Math.max(SKILL_MIN, skill)) / SKILL_MAX; // 0 at skill 100, 1 at skill 0
  return {
    delaySteps: Math.round(18 * h), // up to 150 ms
    stickNoise: 0.5 * h, // up to ~29 deg sd
    mistakeRate: 0.6 * h, // up to one mistake every ~1.7 s
    mistakeSteps: 30, // 0.25 s
  };
}

export function describeHandicap(skill: number): string {
  const c = handicapFor(skill);
  if (c.delaySteps === 0 && c.stickNoise === 0 && c.mistakeRate === 0) return "no handicap";
  return `${Math.round((c.delaySteps / 120) * 1000)} ms delay, ±${Math.round((c.stickNoise * 180) / Math.PI)}° stick noise, ${c.mistakeRate.toFixed(2)} mistakes/s`;
}
