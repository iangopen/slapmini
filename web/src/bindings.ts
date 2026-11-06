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
  /** UNVERIFIED GUESS: slow walk while held (sim input PlayerInput.brake; the real game's braking is unmeasured). */
  brake: "Space",
  // The raised stick is a MOUSE button: see MOUSE_BINDINGS below.

  // Practice tools. APPROXIMATE semantics, written from patch notes, not checked in the real game.
  /** Set puck spawn at cursor; hold to raise spawn height; release drops the puck from the spawn. */
  spawn: "KeyF",
  /** Set a puck target marker at the cursor. */
  target: "KeyG",
  /** Reset the puck to its spawn point, at rest. */
  reset: "KeyR",
  /** Call the puck toward the stick's forehand. */
  call: "KeyQ",

  /** Game page: open or close the tuning panel. */
  tune: "F2",

  // Match mode.
  /** Start a new match after game over (rematch). */
  rematch: "Enter",
} as const;

/** Mouse buttons (MouseEvent.button). UNVERIFIED GUESS like the keys: the real game raises the stick with the left
 * button (Ian), but its exact behaviour (hold vs toggle, timing) is unmeasured. */
export const MOUSE_BINDINGS = {
  /** Raise the stick while held (w4). In relative mode the click that captures the pointer does not raise. */
  stickUp: 0,
} as const;

/** Extra codes accepted for the same action (e.g. right Shift). */
export const ALIASES: Partial<Record<keyof typeof BINDINGS, string[]>> = {
  dash: ["ShiftRight"],
  rematch: ["NumpadEnter"],
};

/** Key events that belong to a focused field instead of the game. Text fields keep every key; number fields (the tuning
 * panel) only the keys they can use, so WASD, Shift and Space still drive the player right after an edit. */
export function fieldOwnsKey(e: KeyboardEvent): boolean {
  const el = e.target as HTMLElement | null;
  if (!el) return false;
  if (el.tagName === "TEXTAREA" || el.isContentEditable) return true;
  if (el.tagName !== "INPUT") return false;
  const type = (el as HTMLInputElement).type;
  if (type === "number") return e.code === "KeyE" || e.code === "Enter" || e.code === "NumpadEnter";
  return ["text", "search", "email", "password", "url"].includes(type);
}

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
  ["Space", "slow walk while held: WASD still steers (binding unverified guess; speeds are placeholders)"],
  ["Left mouse", "raise the stick while held: a puck on the ice passes under it (unverified; heights are placeholders)"],
  ["F", "set puck spawn at cursor, hold to raise height (approx.)"],
  ["G", "set puck target at cursor (approx.)"],
  ["R", "reset puck to spawn (approx.)"],
  ["Q", "call puck to stick forehand (approx.)"],
  ["Enter", "rematch after game over (match mode)"],
];
