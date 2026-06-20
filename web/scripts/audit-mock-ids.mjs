// One-off audit: verify every hand-typed mlbId in the web/ mock data against
// MLB's official Stats API. Catches wrong-headshot bugs (id maps to a different
// player than the mock's name). Scratch tool — not committed.
import { readFileSync } from "node:fs";

const FILES = [
  "web/src/lib/players-data.ts",
  "web/src/lib/research-data.ts",
  "web/src/lib/optimizer-data.ts",
  "web/src/lib/trades-data.ts",
  "web/src/lib/data.ts",
  "web/src/lib/streaming-data.ts",
];

// Extract the player NAME from a row, trying each mock shape in order.
function extractName(line) {
  let m;
  if ((m = line.match(/\bp\(\s*"([^"]+)"/))) return m[1]; // optimizer p("Name",...)
  if ((m = line.match(/\btp\(\s*"([^"]+)"/))) return m[1]; // trades   tp("Name",...)
  if ((m = line.match(/\b(?:fa|lr)\(\s*\d+\s*,\s*"([^"]+)"/))) return m[1]; // fa/lr(rank,"Name",...)
  if ((m = line.match(/\b(?:pr|ref)\(\s*"([^"]+)"/))) return m[1]; // streaming pr("Name",...) / closers ref("Name",...)
  if ((m = line.match(/name:\s*"([^"]+)"/))) return m[1]; // data.ts  { name: "Name", ... }
  if ((m = line.match(/"([^"]+)"/))) return m[1]; // fallback: first quoted string
  return null;
}

const pairs = new Map(); // mlbId -> { name, where }
for (const f of FILES) {
  const text = readFileSync(f, "utf8");
  text.split("\n").forEach((line, i) => {
    const idm = line.match(/\b(\d{6})\b/);
    if (!idm) return;
    const name = extractName(line);
    if (!name) return;
    const id = idm[1];
    if (!pairs.has(id)) pairs.set(id, { name, where: `${f}:${i + 1}` });
  });
}

const norm = (s) =>
  s
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "") // strip accents
    .toLowerCase()
    .replace(/[.\-']/g, "")
    .replace(/\b(jr|sr|ii|iii|iv)\b/g, "")
    .replace(/\s+/g, " ")
    .trim();

const results = [];
for (const [id, info] of pairs) {
  try {
    const r = await fetch(`https://statsapi.mlb.com/api/v1/people/${id}`);
    const j = await r.json();
    const p = j?.people?.[0];
    const real = p?.fullName ?? "(NOT FOUND)";
    const team = p?.currentTeam?.name ?? "?";
    const a = norm(real);
    const b = norm(info.name);
    const ok = a === b || (a && b && (a.includes(b) || b.includes(a)));
    results.push({ id, mock: info.name, real, team, ok, where: info.where });
  } catch (e) {
    results.push({ id, mock: info.name, real: `ERROR ${e.message}`, team: "?", ok: false, where: info.where });
  }
}

const bad = results.filter((r) => !r.ok);
console.log(`Checked ${results.length} unique mlbIds across ${FILES.length} mock files.\n`);
console.log(`=== MISMATCHES (${bad.length}) ===`);
for (const r of bad) console.log(`  ${r.id}  mock="${r.mock}"  ->  real="${r.real}" (${r.team})  @ ${r.where}`);
console.log(`\n=== OK (${results.length - bad.length}) ===`);
for (const r of results.filter((x) => x.ok)) console.log(`  ${r.id}  ${r.mock} ✓`);
