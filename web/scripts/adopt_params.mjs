// Writes the values of an exported tuning-panel JSON into the repo-root params.json.
//   npm run adopt-params -- path/to/exported.json [--dry-run]
// Changed rows get provenance `tuned-by-feel` and a dated note; every other byte of params.json stays. Nothing is written
// if any name is unknown or any value is out of range. The physics stamp changes with the values, so training
// resume then needs --allow-physics-change.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const paramsPath = process.env.ADOPT_PARAMS_TARGET ?? path.join(root, "params.json");
const args = process.argv.slice(2);
const dry = args.includes("--dry-run");
const input = args.find((a) => !a.startsWith("--"));
const die = (m) => { console.error(`adopt-params: ${m}`); process.exit(1); };
if (!input) die("usage: npm run adopt-params -- exported.json [--dry-run]");

const text = readFileSync(paramsPath, "utf8");
const current = JSON.parse(text).params;
let exported;
try { exported = JSON.parse(readFileSync(input, "utf8")); } catch (e) { die(`cannot read ${input}: ${e.message}`); }
const rows = exported.params ?? exported; // a panel export, or a plain { name: value } map
const valueOf = (e) => (typeof e === "number" ? e : e?.value);

// Stick chain points are signed coordinates; everything else in params.json is a non-negative quantity.
const signed = (n) => /^stick_(hand|kink|heel|toe)_[xy]$/.test(n);
function rangeError(n, v, old) {
  if (typeof v !== "number" || !Number.isFinite(v)) return "not a finite number";
  if (!signed(n) && v < 0) return "negative";
  if ((n === "timestep" || n === "substeps") && v <= 0) return "must be positive";
  if (n === "stick_hit_rule" && v !== 0 && v !== 1) return "must be 0 or 1";
  const limit = Math.max(10 * Math.abs(old), 1);
  if (Math.abs(v) > limit) return `more than 10x the current value (limit ${limit})`;
  return null;
}

const errors = [], changes = [];
for (const [n, e] of Object.entries(rows)) {
  if (n.startsWith("_")) continue;
  if (!(n in current)) { errors.push(`unknown name ${n}`); continue; }
  const v = valueOf(e), old = current[n].value;
  const bad = rangeError(n, v, old);
  if (bad) errors.push(`${n} = ${v}: ${bad}`);
  else if (v !== old) changes.push({ n, old, v, was: current[n].provenance });
}
if (errors.length) die(`nothing written:\n  ${errors.join("\n  ")}`);
if (!changes.length) { console.log("adopt-params: no value differs from params.json"); process.exit(0); }

const date = new Date().toISOString().slice(0, 10);
let out = text;
for (const c of changes) {
  const was = `was ${c.old}, ${c.was}`;
  const line = new RegExp(`^(\\s*"${c.n}":\\s*)\\{.*?\\}(,?)(\\r?)$`, "m");
  const m = out.match(line);
  if (!m) die(`${c.n}: its row is not on one line`);
  const unit = current[c.n].unit;
  const entry = { value: c.v, provenance: "tuned-by-feel", ...(unit ? { unit } : {}), note: `tuned-by-feel ${date} (${was}): adopted from the tuning panel.` };
  out = out.replace(line, () => `${m[1]}${JSON.stringify(entry).replace(/":/g, '": ').replace(/,"/g, ', "')}${m[2]}${m[3]}`);
}
JSON.parse(out);
console.log(`${dry ? "would change" : "changed"} ${changes.length} of ${Object.keys(current).length} params in ${path.relative(process.cwd(), paramsPath)}:`);
for (const c of changes) console.log(`- ${c.n}: ${c.old} (${c.was})\n+ ${c.n}: ${c.v} (tuned-by-feel ${date})`);
if (!dry) writeFileSync(paramsPath, out);
