"use client";

import { useEffect, useMemo, useState } from "react";
import { Microscope } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { StreamScorecard } from "@/components/viz/StreamScorecard";
import { analyzePitcher, formatStreamDate, type ProbableStarter, type PosGroup, type PitcherScorecard } from "@/lib/streaming-data";

const GROUPS: (PosGroup | "All")[] = ["All", "SP", "SP/RP", "RP"];

export function AnalyzeStarter({ probables, date }: { probables: ProbableStarter[]; date: string }) {
  const [group, setGroup] = useState<PosGroup | "All">("All");
  const list = useMemo(
    () => (group === "All" ? probables : probables.filter((p) => p.posGroup === group)),
    [group, probables],
  );
  const [mlbId, setMlbId] = useState<number>(probables[0]?.player.mlbId ?? 0);
  const selected = list.find((p) => p.player.mlbId === mlbId) ?? list[0];

  // Analyze is async (live = POST /api/streaming/analyze). Re-run on selection/date change.
  const [card, setCard] = useState<PitcherScorecard | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const selectedMlbId = selected?.player.mlbId;
  useEffect(() => {
    let alive = true;
    // setState only inside async callbacks (never synchronously in the effect body).
    Promise.resolve()
      .then(() => {
        if (!alive) return undefined;
        setLoading(true);
        setFailed(false);
        return selected ? analyzePitcher(selected, date) : null;
      })
      .then((c) => {
        if (!alive || c === undefined) return;
        setCard(c);
        setLoading(false);
      })
      .catch(() => {
        // Live: analyzePitcher can now reject (HIGH-3) — show an inline error
        // instead of a fabricated scorecard or a stuck "Scoring…" spinner.
        if (!alive) return;
        setCard(null);
        setLoading(false);
        setFailed(true);
      });
    return () => {
      alive = false;
    };
    // selected resolves from selectedMlbId + the stable list; keying on the id avoids a refetch loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedMlbId, date]);

  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <Microscope className="size-4 text-heat" aria-hidden /> Analyze any starter
      </div>
      <p className="mb-3 text-[12px] text-ink-3">
        Score any probable starter in MLB for this date — rostered or not — through the streaming algorithm.
      </p>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex gap-1" role="group" aria-label="Filter by position group">
          {GROUPS.map((g) => (
            <button
              key={g}
              onClick={() => setGroup(g)}
              aria-pressed={group === g}
              className={`rounded-lg px-2.5 py-1 text-[12px] font-bold ${group === g ? "bg-navy text-white" : "bg-surface text-ink-2 hover:bg-surface-2"}`}
            >
              {g}
            </button>
          ))}
        </div>
        <label htmlFor="analyze-starter-select" className="sr-only">
          Select a probable starter to analyze
        </label>
        <select
          id="analyze-starter-select"
          value={selected?.player.mlbId ?? 0}
          onChange={(e) => setMlbId(Number(e.target.value))}
          className="min-h-9 flex-1 rounded-lg border border-line bg-canvas px-3 text-sm font-semibold text-navy"
        >
          {list.map((p) => (
            <option key={p.player.mlbId} value={p.player.mlbId}>
              {p.player.name} ({p.team}) {p.isHome ? "vs" : "@"} {p.opponent} · {p.startLikelihood}
            </option>
          ))}
        </select>
      </div>

      {loading && <div className="mt-5 text-[13px] text-ink-3">Scoring…</div>}
      {!loading && failed && (
        <div className="mt-5 text-[13px] text-ember">Couldn&apos;t reach the scorer. Try again.</div>
      )}
      {!loading && !failed && !card && selected && (
        <div className="mt-5 text-[13px] text-ink-3">Couldn&apos;t score this starter for {formatStreamDate(date)}.</div>
      )}
      {!loading && card && selected && (
        <div className="mt-5 flex flex-wrap items-start gap-6">
          <StreamScorecard score={card.score} components={card.components} size={168} />
          <div className="min-w-0 flex-1 space-y-3">
            <div className="flex items-center gap-3">
              <PlayerAvatar mlbId={selected.player.mlbId} teamId={selected.player.teamId} name={selected.player.name} size={44} />
              <div>
                <div className="font-display text-lg font-bold text-navy">{selected.player.name}</div>
                <div className="text-[12px] text-ink-2">{selected.player.pos} · {card.expectedLine}</div>
              </div>
            </div>
            {card.riskFlags.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {card.riskFlags.map((f) => (
                  <span key={f} className="rounded bg-ember/10 px-1.5 py-0.5 text-[11px] font-bold text-ember">{f}</span>
                ))}
              </div>
            )}
            <ul className="space-y-1.5">
              {card.factors.map((fct) => (
                <li key={fct.key} className="flex items-baseline justify-between gap-3 border-b border-line/50 pb-1 text-[12px]">
                  <span className="font-semibold text-navy">{fct.label}</span>
                  <span className="flex-1 truncate px-2 text-ink-3">{fct.detail}</span>
                  <span className={`tnum font-bold ${fct.value >= 0 ? "text-ok" : "text-ember"}`}>{fct.value >= 0 ? "+" : ""}{fct.value.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </Card>
  );
}
