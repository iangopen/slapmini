// All key bindings in one place. Values are KeyboardEvent.code strings.

export const BINDINGS = {
  up: "KeyW",
  down: "KeyS",
  left: "KeyA",
  right: "KeyD",
  /** UNVERIFIED GUESS: the real game's dash binding has not been checked. */
  dash: "ShiftLeft",
  /** UNVERIFIED GUESS: the real game's lift binding has not been checked (was Space before w3). */
  lift: "KeyE",
  /** UNVERIFIED GUESS: stop / brake (web-layer brake, see brake.ts; the real game's braking is unmeasured). */
  brake: "Space",

  // Practice tools. APPROXIMATE semantics, written from patch notes, not checked in the real game.
  /** Set puck spawn at cursor; hold to raise spawn height; release drops the puck from the spawn. */
  spawn: "KeyF",
  /** Set a puck target marker at the cursor. */
  target: "KeyG",
  /** Reset the puck to its spawn point, at rest. */
  reset: "KeyR",
  /** Call the puck toward the stick's forehand. */
  call: "KeyQ",

  // Match mode.
  /** Start a new match after game over (rematch). */
  rematch: "Enter",
} as const;

/** Extra codes accepted for the same action (e.g. right Shift). */
export const ALIASES: Partial<Record<keyof typeof BINDINGS, string[]>> = {
  dash: ["ShiftRight"],
  rematch: ["NumpadEnter"],
};

export function actionFor(code: string): keyof typeof BINDINGS | undefined {
  for (const [k, v] of Object.entries(BINDINGS)) if (v === code) return k as keyof typeof BINDINGS;
  for (const [k, v] of Object.entries(ALIASES)) if (v?.includes(code)) return k as keyof typeof BINDINGS;
  return undefined;
}

export const KEY_HELP: [string, string][] = [
  ["WASD", "move"],
  ["Mouse", "stick target angle"],
  ["Shift", "dash (binding unverified guess)"],
  ["E", "lift (binding unverified guess)"],
  ["Space", "brake / stop (binding unverified guess; stopping time is a placeholder)"],
  ["F", "set puck spawn at cursor, hold to raise height (approx.)"],
  ["G", "set puck target at cursor (approx.)"],
  ["R", "reset puck to spawn (approx.)"],
  ["Q", "call puck to stick forehand (approx.)"],
  ["Enter", "rematch after game over (match mode)"],
];
