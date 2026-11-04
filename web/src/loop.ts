// Fixed-timestep driver. Knows nothing about rendering or the DOM, so it can run headless.

export class FixedStepLoop {
  private acc = 0;
  /** Steps dropped because catch-up was capped (e.g. after a tab switch). */
  dropped = 0;

  constructor(
    private readonly stepOnce: () => void,
    private readonly timestep: () => number,
    readonly maxStepsPerAdvance = 5,
  ) {}

  /** Feed real elapsed seconds; runs whole steps and returns how many ran. */
  advance(elapsedSec: number): number {
    const dt = this.timestep();
    this.acc += Math.max(0, elapsedSec);
    let n = 0;
    while (this.acc >= dt && n < this.maxStepsPerAdvance) {
      this.stepOnce();
      this.acc -= dt;
      n++;
    }
    if (this.acc >= dt) {
      // Cap reached: drop the backlog instead of spiraling.
      this.dropped += Math.floor(this.acc / dt);
      this.acc %= dt;
    }
    return n;
  }

  /** Fraction of a step accumulated but not yet simulated (for optional interpolation). */
  get alpha(): number {
    return this.acc / this.timestep();
  }
}
