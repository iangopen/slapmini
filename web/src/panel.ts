// Tuning panel, shared by the lab and the game (F2): one numeric input per param, provenance badge, feel group on top,
// Reset / Export / Import. Edits are experiments: they keep the provenance tag, changed rows are highlighted, and they
// are stored in localStorage as an overlay on params.json, so an edit made in the lab is also there in the game.

import type { ParamFile } from "./types";

export function download(name: string, text: string, type = "application/json"): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const EDITS_KEY = "slapmini.params.edits.v1";

/** A zero or negative timestep would hang the fixed-step loop; everything else is the tuner's experiment. */
function validValue(name: string, v: unknown): boolean {
  return typeof v === "number" && Number.isFinite(v) && (name !== "timestep" || v > 0);
}

/** The stored overlay applied to `original` (a copy). Unknown names and bad values are ignored. */
export function withStoredEdits(original: ParamFile): ParamFile {
  const f = structuredClone(original);
  try {
    const e = JSON.parse(localStorage.getItem(EDITS_KEY) ?? "{}") as Record<string, unknown>;
    for (const [k, v] of Object.entries(e)) if (f.params[k] && validValue(k, v)) f.params[k].value = v as number;
  } catch {
    /* storage blocked or corrupt: no edits */
  }
  return f;
}

function storeEdits(original: ParamFile, f: ParamFile): void {
  const e: Record<string, number> = {};
  for (const [k, v] of Object.entries(f.params)) if (v.value !== original.params[k]?.value) e[k] = v.value;
  try {
    if (Object.keys(e).length) localStorage.setItem(EDITS_KEY, JSON.stringify(e));
    else localStorage.removeItem(EDITS_KEY);
  } catch {
    /* storage blocked: this page only */
  }
}

/** Parse an exported params file against `original`: every name must be known; values only, tags stay as they are. */
export function parseImport(text: string, original: ParamFile): { values: Record<string, number> } | { error: string } {
  let raw: { params?: Record<string, { value?: unknown }> };
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { error: `not JSON: ${(e as Error).message}` };
  }
  if (!raw.params || typeof raw.params !== "object") return { error: "no params object" };
  const values: Record<string, number> = {};
  for (const [k, e] of Object.entries(raw.params)) {
    if (!original.params[k]) return { error: `unknown parameter ${k}` };
    if (!validValue(k, e?.value)) return { error: `bad value for ${k}` };
    values[k] = e.value as number;
  }
  return { values };
}

/** Feel parameters: first in the panel, with a short note on show. */
const FEEL: [string, string][] = [
  ["player_accel", "Speeding up along the pressed key. Higher = quicker to top speed."],
  ["player_turn_accel", "How hard the keys remove sideways velocity. Higher = sharper turns, less slide."],
  ["player_brake_accel", "How hard an opposite key stops you. Higher = quicker stops and reversals."],
  ["player_coast_rate", "Glide decay with no key (1/s). Higher = shorter glide; speed halves in 0.69 / rate s."],
  ["player_max_speed", "Top speed with a key held."],
  ["brake_speed", "Slow walk speed while Space is held."],
  ["brake_decel", "How fast Space sheds speed above the slow walk."],
];

export interface PanelHost {
  original: ParamFile;
  /** The live file. */
  current: () => ParamFile;
  /** Apply a whole file to the running sim and everything that reads the params. */
  applyFile: (f: ParamFile) => void;
}

/** Builds the panel into `root`; returns a refresh function. */
export function buildTuningPanel(root: HTMLElement, host: PanelHost): () => void {
  const { original } = host;
  const rows = new Map<string, { row: HTMLElement; input: HTMLInputElement }>();
  root.innerHTML = "";
  const bar = document.createElement("div");
  bar.className = "row tune-bar";
  const mk = (id: string, text: string) => {
    const b = document.createElement("button");
    b.id = id;
    b.textContent = text;
    bar.append(b);
    return b;
  };
  const reset = mk("params-reset", "Reset to file");
  const exp = mk("params-export", "Export JSON");
  const imp = mk("params-import", "Import JSON");
  const file = document.createElement("input");
  file.type = "file";
  file.accept = ".json,application/json";
  file.id = "params-import-file";
  file.hidden = true;
  const status = document.createElement("div");
  status.id = "params-status";
  status.className = "muted small";
  status.setAttribute("role", "status");
  root.append(bar, file, status);

  const refresh = () => {
    const f = host.current();
    let changed = 0;
    for (const [name, { row, input }] of rows) {
      if (document.activeElement !== input) input.value = String(f.params[name].value);
      const c = f.params[name].value !== original.params[name].value;
      row.classList.toggle("changed", c);
      if (c) changed++;
    }
    status.textContent = changed ? `${changed} changed from params.json (an experiment, kept in this browser until Reset)` : "All values as in params.json";
  };
  const commit = (f: ParamFile) => {
    host.applyFile(f);
    storeEdits(original, f);
    refresh();
  };

  const addRow = (name: string, feelNote?: string) => {
    const e = original.params[name];
    const row = document.createElement("div");
    row.className = "param";
    row.title = [e.unit && `unit: ${e.unit}`, e.note].filter(Boolean).join("\n");
    const label = document.createElement("label");
    label.textContent = name;
    label.htmlFor = `param-${name}`;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.id = `param-${name}`;
    input.dataset.param = name;
    input.value = String(e.value);
    input.addEventListener("input", () => {
      const v = Number(input.value);
      if (input.value.trim() === "" || !validValue(name, v)) return;
      const f = structuredClone(host.current());
      f.params[name].value = v;
      commit(f);
    });
    // Enter = done. Leaving the field shows the applied value again.
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") input.blur();
    });
    input.addEventListener("blur", refresh);
    const badge = document.createElement("span");
    badge.className = `badge ${e.provenance}`;
    badge.textContent = e.provenance;
    row.append(label, input, badge);
    root.append(row);
    if (feelNote) {
      const n = document.createElement("div");
      n.className = "param-note muted small";
      n.textContent = feelNote;
      root.append(n);
    }
    rows.set(name, { row, input });
  };

  const head = (t: string) => {
    const h = document.createElement("h3");
    h.className = "tune-head";
    h.textContent = t;
    root.append(h);
  };
  const feel = FEEL.filter(([n]) => original.params[n]);
  head("Skating feel");
  for (const [n, note] of feel) addRow(n, note);
  head("All parameters");
  for (const name of Object.keys(original.params)) if (!feel.some(([n]) => n === name)) addRow(name);

  reset.addEventListener("click", () => commit(structuredClone(original)));
  exp.addEventListener("click", () => download("params.json", JSON.stringify(host.current(), null, 2) + "\n"));
  imp.addEventListener("click", () => file.click());
  file.addEventListener("change", async () => {
    const t = await file.files?.[0]?.text();
    file.value = "";
    if (t === undefined) return;
    const r = parseImport(t, original);
    if ("error" in r) {
      status.textContent = `Import refused: ${r.error}`;
      return;
    }
    const f = structuredClone(host.current());
    for (const [k, v] of Object.entries(r.values)) f.params[k].value = v;
    commit(f);
    status.textContent = `Imported ${Object.keys(r.values).length} values. ${status.textContent}`;
  });
  refresh();
  return refresh;
}

/** A running match keeps its timers in ticks, set when the phase began. Move them to edited params, so a change of
 * match_length_s, countdown_s, goal_pause_s or timestep shows at once. Returns true if a timer changed. */
export function retimeMatch(
  m: { phase: string; phase_ticks: number; clock_ticks: number; overtime: boolean } | null | undefined,
  a: Record<string, number>,
  b: Record<string, number>,
): boolean {
  if (!m || m.phase === "finished") return false;
  const was = [m.clock_ticks, m.phase_ticks].join();
  const ticks = (s: number) => Math.round(s / b.timestep);
  const scale = a.timestep / b.timestep;
  m.clock_ticks = Math.round(m.clock_ticks * scale);
  m.phase_ticks = Math.round(m.phase_ticks * scale);
  if (!m.overtime) m.clock_ticks = Math.max(1, m.clock_ticks + ticks(b.match_length_s - a.match_length_s));
  if (m.phase === "countdown") m.phase_ticks = Math.max(1, m.phase_ticks + ticks(b.countdown_s - a.countdown_s));
  if (m.phase === "goal_pause") m.phase_ticks = Math.max(1, m.phase_ticks + ticks(b.goal_pause_s - a.goal_pause_s));
  return was !== [m.clock_ticks, m.phase_ticks].join();
}
