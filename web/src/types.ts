// Shapes mirror the Rust serde types in sim/src/state.rs, sim/src/params.rs and calib/src/lib.rs.

export interface PlayerInput {
  move_x: number;
  move_y: number;
  stick_target_angle: number;
  dash: boolean;
  lift: boolean;
}

export interface Puck {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  in_net: number | null;
}

export interface Player {
  team: number;
  x: number; y: number; vx: number; vy: number;
  facing: number;
  stick_angle: number;
  stick_omega: number;
  dash_timer: number;
  dash_cooldown: number;
  dash_prev: boolean;
}

export interface State {
  tick: number;
  puck: Puck;
  players: Player[];
  score: number[];
}

export interface ParamEntry { value: number; provenance: string; unit?: string; note?: string }
export interface ParamFile { _comment?: string; params: Record<string, ParamEntry> }

/** calib::InputEvent: input held from `step` until that player's next event. */
export interface InputEvent { step: number; player: number; input: PlayerInput }

/** calib::Scenario (`*.scenario.json`). */
export interface Scenario {
  name: string;
  description: string;
  steps: number;
  initial_state: State;
  inputs: InputEvent[];
}

/** One row of a calib trajectory CSV. */
export interface Sample { t: number; x: number; y: number; z: number; stick_angle?: number }

/** Flat numeric view of params used by the renderer and tools. */
export type P = Record<string, number>;

export function flatParams(f: ParamFile): P {
  const out: P = {};
  for (const [k, e] of Object.entries(f.params)) out[k] = e.value;
  return out;
}
