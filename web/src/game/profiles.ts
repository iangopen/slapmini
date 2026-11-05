// Opponent profiles: web/public/bots/index.json lists profile files; each profile is
//   { "name", "description", "policy": "file.onnx" | null, "scripted": "chaser" | ... | null,
//     "style": [floats, up to 8], "skill": 0..100, "fallback": "chaser", "smoke_test": bool }
// exactly one of policy / scripted is set. Paths are relative to the bots/ folder.
// To add a trained bot: drop NAME.onnx (+ its .json sidecar from train.export_onnx) into
// web/public/bots/, write NAME.profile.json, and add it to index.json.

export interface Profile {
  id: string;
  name: string;
  description: string;
  policy: string | null;
  scripted: string | null;
  style: number[];
  skill: number;
  fallback: string;
  smoke_test?: boolean;
}

export const SCRIPTED = ["chaser", "defender", "cautious", "idle"] as const;
export const BOTS_DIR = "bots/";

/** Built-in scripted opponents, available even if index.json is missing. */
export const BUILTIN: Profile[] = [
  { id: "chaser", name: "Chaser", description: "Scripted: skates to the puck and shoots at the net.", policy: null, scripted: "chaser", style: [], skill: 100, fallback: "chaser" },
  { id: "defender", name: "Defender", description: "Scripted: guards its net and clears the puck. Rarely attacks.", policy: null, scripted: "defender", style: [], skill: 100, fallback: "chaser" },
  { id: "cautious", name: "Cautious", description: "Scripted: waits in its half and only attacks a puck that comes to it.", policy: null, scripted: "cautious", style: [], skill: 100, fallback: "chaser" },
];

export function validate(p: Partial<Profile>, id: string): Profile {
  if (!p.name) throw new Error(`profile ${id}: missing name`);
  if (!!p.policy === !!p.scripted) throw new Error(`profile ${id}: set exactly one of policy / scripted`);
  if (p.scripted && !SCRIPTED.includes(p.scripted as (typeof SCRIPTED)[number])) throw new Error(`profile ${id}: unknown scripted bot ${p.scripted}`);
  return {
    id,
    name: p.name,
    description: p.description ?? "",
    policy: p.policy ?? null,
    scripted: p.scripted ?? null,
    style: (p.style ?? []).slice(0, 8).map(Number),
    skill: Math.min(100, Math.max(0, Number(p.skill ?? 100))),
    fallback: SCRIPTED.includes(p.fallback as (typeof SCRIPTED)[number]) ? p.fallback! : "chaser",
    smoke_test: !!p.smoke_test,
  };
}

/** All opponents: index.json entries plus the built-in scripted bots not listed there. Errors are returned, not thrown. */
export async function loadProfiles(base = BOTS_DIR): Promise<{ profiles: Profile[]; errors: string[] }> {
  const errors: string[] = [];
  const out: Profile[] = [];
  try {
    const idx = await (await fetch(base + "index.json")).json();
    for (const file of idx.profiles as string[]) {
      try {
        const r = await fetch(base + file);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        out.push(validate(await r.json(), file.replace(/\.profile\.json$|\.json$/, "")));
      } catch (e) {
        errors.push(`${file}: ${(e as Error).message}`);
      }
    }
  } catch (e) {
    errors.push(`bots/index.json: ${(e as Error).message}`);
  }
  for (const b of BUILTIN) if (!out.some((p) => p.scripted === b.scripted && !p.policy)) out.push(b);
  return { profiles: out, errors };
}
