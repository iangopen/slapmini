// Param panel: one numeric input per param, provenance badge, reset-to-file and export.

import type { ParamFile } from "./types";

export function download(name: string, text: string, type = "application/json"): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function buildParamPanel(
  root: HTMLElement,
  original: ParamFile,
  current: () => ParamFile,
  apply: (name: string, value: number) => void,
): () => void {
  const rows = new Map<string, { row: HTMLElement; input: HTMLInputElement }>();
  root.innerHTML = "";
  for (const [name, e] of Object.entries(original.params)) {
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
      if (input.value.trim() === "" || !Number.isFinite(v)) return;
      apply(name, v);
      row.classList.toggle("changed", v !== original.params[name].value);
    });
    const badge = document.createElement("span");
    badge.className = `badge ${e.provenance}`;
    badge.textContent = e.provenance;
    row.append(label, input, badge);
    root.append(row);
    rows.set(name, { row, input });
  }
  // Refresh inputs from the live file (after reset).
  return () => {
    const f = current();
    for (const [name, { row, input }] of rows) {
      input.value = String(f.params[name].value);
      row.classList.toggle("changed", f.params[name].value !== original.params[name].value);
    }
  };
}
