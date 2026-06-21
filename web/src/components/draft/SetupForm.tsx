"use client";

import { useState } from "react";
import { Swords } from "lucide-react";
import { Card } from "@/components/ui/Card";
import type { DraftConfig } from "@/lib/draft-data";

/** Pre-draft setup: league size, rounds, and your snake seat (1-based in the UI,
 *  converted to 0-based userTeamIndex on submit). */
export function SetupForm({ onStart, busy }: { onStart: (c: DraftConfig) => void; busy: boolean }) {
  const [numTeams, setNumTeams] = useState(12);
  const [numRounds, setNumRounds] = useState(23);
  const [position, setPosition] = useState(1);

  const clampedPos = Math.min(Math.max(1, position), numTeams);
  const submit = () =>
    onStart({ numTeams, numRounds, userTeamIndex: clampedPos - 1 });

  return (
    <div className="mx-auto max-w-lg">
      <div className="mb-6 text-center">
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
          Preseason · Scouting
        </div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Draft Simulator</h1>
        <p className="mt-1 text-[13px] text-ink-2">
          Mock a snake draft against AI opponents with Monte-Carlo recommendations on every pick.
        </p>
      </div>
      <Card className="p-6">
        <div className="grid grid-cols-2 gap-4">
          <NumField label="Teams" value={numTeams} min={6} max={20} onChange={setNumTeams} />
          <NumField label="Rounds" value={numRounds} min={10} max={30} onChange={setNumRounds} />
        </div>
        <div className="mt-4">
          <NumField
            label={`Your draft position (1–${numTeams})`}
            value={clampedPos}
            min={1}
            max={numTeams}
            onChange={setPosition}
          />
        </div>
        <button
          onClick={submit}
          disabled={busy}
          className="mt-6 inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-b from-heat-bright to-heat px-5 py-3 text-sm font-bold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.01] active:scale-95 disabled:opacity-60 motion-reduce:transform-none"
        >
          <Swords className="size-4" aria-hidden />
          {busy ? "Setting the board…" : "Start Mock Draft"}
        </button>
      </Card>
    </div>
  );
}

function NumField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (n: number) => void;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-bold uppercase tracking-wide text-ink-3">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (!Number.isNaN(n)) onChange(Math.min(max, Math.max(min, Math.round(n))));
        }}
        className="tnum h-11 w-full rounded-lg border border-line bg-canvas px-3 text-[15px] font-bold text-navy outline-none focus:border-heat focus:ring-2 focus:ring-heat/30"
      />
    </label>
  );
}
