// Mouse → stick target, shared by the game and the lab.
//
// "relative" (default, like the real game): the canvas captures the pointer (Pointer Lock API, first
// click), and only HORIZONTAL mouse movement turns the stick: moving right swings it clockwise on screen
// (to the right of where it points), left counterclockwise; vertical movement is ignored. The angle is
// kept in SCREEN space, starts pointing forward (toward the opponent's net) and goes to the sim as
// stick_target_angle in world radians through the view transform (render.ts). It is never clamped (the
// real game's limits are unknown). The stick then turns only as fast as the sim's chase limits allow.
// "cursor": the old model, the stick points from the player toward the cursor.
//
// Settings live in localStorage ("slapmini.mouse.v1"), read and written inside try/catch.

import { screenAngleToWorld, worldAngleToScreen, type View } from "./render";

export type MouseMode = "relative" | "cursor";
export interface MouseSettings {
  mode: MouseMode;
  /** Radians of stick turn per unit of movementX (CSS pixels of mouse movement). */
  sensitivity: number;
}

export const SENS_MIN = 0.001;
export const SENS_MAX = 0.02;
export const SENS_STEP = 0.0005;
export const DEFAULT_SETTINGS: MouseSettings = { mode: "relative", sensitivity: 0.005 };
const KEY = "slapmini.mouse.v1";

export function loadSettings(): MouseSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const s = JSON.parse(raw);
      const mode: MouseMode = s.mode === "cursor" ? "cursor" : "relative";
      const sens = Number(s.sensitivity);
      return { mode, sensitivity: Number.isFinite(sens) ? Math.min(SENS_MAX, Math.max(SENS_MIN, sens)) : DEFAULT_SETTINGS.sensitivity };
    }
  } catch {
    /* storage blocked or corrupt: defaults */
  }
  return { ...DEFAULT_SETTINGS };
}

export function saveSettings(s: MouseSettings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* storage blocked: settings last for this page only */
  }
}

/**
 * Maximum lead of the relative aim over the stick's actual angle (radians). The sim's stick chase turns
 * the short way (wrap(target − stick)) at ≤ 26.17 rad/s; a target more than π ahead flips the short way,
 * so the stick reversed and stalled near 180° when the mouse outran it (w3 reproduction: a fast flick
 * turned the stick 162° of 3438°, reversing for 60 steps). Well under π, the short way can never flip, and
 * the mouse movement beyond the lead is dropped instead of piling up as a debt the stick works off later.
 */
export const MAX_LEAD = 1.2;

const wrapAngle = (a: number): number => {
  const t = (a + Math.PI) % (2 * Math.PI);
  return (t < 0 ? t + 2 * Math.PI : t) - Math.PI;
};

/** The relative stick aim of one player, in screen radians (counterclockwise from screen-right, wrapped to (−π, π]). */
export class StickAim {
  screenAngle = Math.PI / 2; // up the screen
  /** Reads (sim steps) since the aim was last pulled back to MAX_LEAD. */
  private sinceClamp = Infinity;
  /** The mouse is outrunning the stick: the aim hit MAX_LEAD within the last 8 sim steps (~67 ms). */
  get clamped(): boolean {
    return this.sinceClamp < 8;
  }
  constructor(public settings: MouseSettings) {}
  /** Point forward along the world angle `forward` (e.g. 0 = toward +x, the net BLUE attacks). */
  reset(v: View, forward: number): void {
    this.screenAngle = wrapAngle(worldAngleToScreen(v, forward));
  }
  /** Horizontal mouse movement: right (dx > 0) turns clockwise on screen. Vertical movement is ignored. */
  move(dx: number, _dy = 0): void {
    if (Number.isFinite(dx)) this.screenAngle = wrapAngle(this.screenAngle - dx * this.settings.sensitivity);
  }
  /**
   * The sim's stick_target_angle (world radians). With `stickWorld` (the sim's current stick_angle) the aim
   * is first pulled to within MAX_LEAD of the stick, in screen space, and stored that way: call this once per
   * sim step with the stick angle. Without it (readouts, tests) the aim is returned unchanged.
   */
  worldAngle(v: View, stickWorld?: number): number {
    if (stickWorld !== undefined && Number.isFinite(stickWorld)) {
      const stickScreen = worldAngleToScreen(v, stickWorld);
      const lead = wrapAngle(this.screenAngle - stickScreen);
      if (Math.abs(lead) > MAX_LEAD) {
        this.screenAngle = wrapAngle(stickScreen + Math.sign(lead) * MAX_LEAD);
        this.sinceClamp = 0;
      } else this.sinceClamp++;
    }
    return screenAngleToWorld(v, this.screenAngle);
  }
}

/**
 * Pointer lock on a canvas. `wanted()` says whether a click should capture (relative mode, in play).
 * `onMove(dx, dy)` gets movementX/Y only while locked. `onChange(locked)` fires on capture / release.
 * Esc is handled by the browser (it releases the lock); `releasedRecently()` lets the caller ignore that
 * same Esc keydown so it doesn't also leave the match.
 */
export class PointerCapture {
  private releasedAt = -1e9;
  constructor(private canvas: HTMLCanvasElement, wanted: () => boolean, onMove: (dx: number, dy: number) => void, onChange: (locked: boolean) => void) {
    canvas.addEventListener("mousedown", () => {
      if (wanted() && !this.locked) this.request();
    });
    document.addEventListener("pointerlockchange", () => {
      if (!this.locked) this.releasedAt = performance.now();
      onChange(this.locked);
    });
    document.addEventListener("pointerlockerror", () => onChange(false));
    document.addEventListener("mousemove", (e) => {
      if (this.locked) onMove(e.movementX, e.movementY);
    });
  }
  get locked(): boolean {
    return document.pointerLockElement === this.canvas;
  }
  request(): void {
    try {
      const r = this.canvas.requestPointerLock() as unknown as Promise<void> | undefined;
      r?.catch?.(() => undefined); // denied (no user gesture, iframe policy): stay unlocked, the hint stays up
    } catch {
      /* not supported */
    }
  }
  release(): void {
    if (this.locked) document.exitPointerLock();
  }
  /** True for ~250 ms after the lock was released (the Esc that released it). */
  releasedRecently(): boolean {
    return performance.now() - this.releasedAt < 250;
  }
}

/**
 * The raised stick (w4): the left mouse button (MOUSE_BINDINGS.stickUp) held = stick_up. In relative mode the click
 * that captures the pointer only captures. The stick is lowered on button up, window blur, pointer unlock and
 * `release()` (match end, leaving the match), so it can never stay stuck up.
 */
export class StickRaise {
  held = false;
  constructor(canvas: HTMLCanvasElement, capture: PointerCapture, relative: () => boolean, active: () => boolean, button: number) {
    // Registered after PointerCapture's own mousedown (which requests the lock): during the capturing click the
    // pointer is not locked yet, so that click is recognised and does not raise.
    canvas.addEventListener("mousedown", (e) => {
      if (e.button !== button || !active()) return;
      if (relative() && !capture.locked) return; // this click only captures the mouse
      this.held = true;
    });
    window.addEventListener("mouseup", (e) => {
      if (e.button === button) this.held = false;
    });
    window.addEventListener("blur", () => (this.held = false));
    document.addEventListener("pointerlockchange", () => {
      if (!capture.locked) this.held = false;
    });
  }
  release(): void {
    this.held = false;
  }
}
