"use client";

import { useEffect, useState } from "react";
import { Search, X, ArrowRight, Scale, TrendingUp, TriangleAlert } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { searchPlayers, type PlayerPick } from "@/lib/player-search";
import { evaluateTrade, type TradeEval } from "@/lib/trades-data";
import { isPaywall } from "@/lib/api/errors";
import { PaywallGate } from "@/components/billing/PaywallGate";
import { cn } from "@/lib/utils";

/**
 * Build-a-trade evaluator — search any player into the You-Give / You-Get baskets,
 * then POST /api/trade/evaluate for the full engine grade. Search is DB-backed
 * (real locally); the receiving-side eval is pool-backed (real), the giving side
 * needs your roster (full on Railway).
 */
export function BuildPanel() {
  const [giving, setGiving] = useState<PlayerPick[]>([]);
  const [receiving, setReceiving] = useState<PlayerPick[]>([]);
  const [result, setResult] = useState<TradeEval | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [locked, setLocked] = useState(false); // 402 paywall on /api/trade/evaluate

  const add = (side: "give" | "get", p: PlayerPick) => {
    (side === "give" ? setGiving : setReceiving)((cur) => (cur.some((x) => x.id === p.id) ? cur : [...cur, p]));
    setResult(null);
    setLocked(false);
  };
  const remove = (side: "give" | "get", id: number) => {
    (side === "give" ? setGiving : setReceiving)((cur) => cur.filter((x) => x.id !== id));
    setResult(null);
    setLocked(false);
  };

  const ready = giving.length > 0 && receiving.length > 0;
  const chosen = new Set([...giving, ...receiving].map((p) => p.id));

  const onEvaluate = () => {
    if (!ready) return;
    setEvaluating(true);
    setLocked(false);
    evaluateTrade(giving, receiving)
      .then(setResult)
      .catch((e) => {
        if (isPaywall(e)) setLocked(true); // signed-in Free user → paywall
      })
      .finally(() => setEvaluating(false));
  };

  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-[1fr_auto_1fr] md:items-start">
        <SearchBasket label="You Give" tone="give" players={giving} chosen={chosen} onAdd={(p) => add("give", p)} onRemove={(id) => remove("give", id)} />
        <div className="hidden items-center justify-center pt-7 md:flex">
          <span className="flex size-9 items-center justify-center rounded-full bg-surface text-ink-3">
            <ArrowRight className="size-5" aria-hidden />
          </span>
        </div>
        <SearchBasket label="You Get" tone="get" players={receiving} chosen={chosen} onAdd={(p) => add("get", p)} onRemove={(id) => remove("get", id)} />
      </div>

      <div className="flex justify-center">
        <button
          onClick={onEvaluate}
          disabled={!ready || evaluating}
          className="inline-flex min-h-10 items-center gap-2 rounded-xl bg-gradient-to-b from-heat-bright to-heat px-5 text-sm font-bold text-white shadow-[0_4px_12px_rgba(255,92,16,0.3)] transition-transform duration-[var(--dur-1)] enabled:hover:scale-[1.02] enabled:active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 motion-reduce:transform-none"
        >
          <Scale className="size-4" aria-hidden />
          {evaluating ? "Evaluating…" : "Evaluate Trade"}
        </button>
      </div>

      {evaluating ? (
        <Skeleton className="h-44 w-full rounded-2xl" />
      ) : locked ? (
        <PaywallGate feature="Trade evaluation" />
      ) : result ? (
        <ResultCard result={result} />
      ) : (
        !ready && (
          <p className="text-center text-[13px] text-ink-3">
            Add at least one player to each side, then evaluate.
          </p>
        )
      )}
    </div>
  );
}

function SearchBasket({
  label,
  tone,
  players,
  chosen,
  onAdd,
  onRemove,
}: {
  label: string;
  tone: "give" | "get";
  players: PlayerPick[];
  chosen: Set<number>;
  onAdd: (p: PlayerPick) => void;
  onRemove: (id: number) => void;
}) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<PlayerPick[]>([]);

  useEffect(() => {
    let alive = true;
    const id = setTimeout(() => {
      searchPlayers(q).then((r) => alive && setResults(r));
    }, 250);
    return () => {
      alive = false;
      clearTimeout(id);
    };
  }, [q]);

  const shown = results.filter((p) => !chosen.has(p.id)).slice(0, 12);

  return (
    <div>
      <div className={cn("mb-1.5 text-[10px] font-bold uppercase tracking-wide", tone === "give" ? "text-ember" : "text-ok")}>
        {label}
      </div>

      <div className="mb-2 space-y-1.5">
        {players.length === 0 && (
          <div className="rounded-lg border border-dashed border-line bg-surface/50 px-3 py-2 text-[12px] text-ink-3">
            No players yet
          </div>
        )}
        {players.map((p) => (
          <div key={p.id} className="flex items-center gap-2 rounded-lg border border-line bg-surface px-2.5 py-1.5">
            <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={26} />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[13px] font-semibold text-navy">{p.name}</span>
              <span className="tnum block text-[10.5px] text-ink-3">
                {p.teamAbbr} · {p.pos}
              </span>
            </span>
            <button
              onClick={() => onRemove(p.id)}
              aria-label={`Remove ${p.name}`}
              className="flex size-6 shrink-0 items-center justify-center rounded-full text-ink-3 transition-colors hover:bg-surface-2 hover:text-ember focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-heat/50"
            >
              <X className="size-3.5" aria-hidden />
            </button>
          </div>
        ))}
      </div>

      <div className="relative">
        <div className="flex items-center gap-2 rounded-lg border border-line bg-surface px-2.5 py-1.5">
          <Search className="size-3.5 text-ink-3" aria-hidden />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search players…"
            className="w-full bg-transparent text-[12.5px] text-navy outline-none placeholder:text-ink-3"
            aria-label={`Search players to ${label.toLowerCase()}`}
          />
        </div>
        {shown.length > 0 && (
          <ul className="absolute z-10 mt-1 max-h-60 w-full overflow-y-auto rounded-lg border border-line bg-canvas shadow-[0_8px_24px_rgba(11,24,48,0.18)]">
            {shown.map((p) => (
              <li key={p.id}>
                <button
                  onClick={() => {
                    onAdd(p);
                    setQ("");
                    setResults([]);
                  }}
                  className="flex w-full items-center gap-2 border-b border-line/60 px-2.5 py-1.5 text-left transition-colors last:border-0 hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-heat/50"
                >
                  <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={24} />
                  <span className="min-w-0">
                    <span className="block truncate text-[12.5px] font-semibold text-navy">{p.name}</span>
                    <span className="tnum block text-[10px] text-ink-3">
                      {p.teamAbbr} · {p.pos}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function gradeBadgeClass(grade: string): string {
  const g = grade[0]?.toUpperCase();
  if (g === "A") return "bg-ok/12 text-ok";
  if (g === "B") return "bg-heat/12 text-heat";
  if (g === "C") return "bg-warn/15 text-warn";
  return "bg-ember/12 text-ember";
}

function ResultCard({ result }: { result: TradeEval }) {
  return (
    <Card className="p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wide text-ink-3">
            <Scale className="size-3.5 text-heat" aria-hidden /> Verdict
          </div>
          <div className="font-display text-lg font-bold text-navy">{result.verdict || "—"}</div>
          <div className="tnum mt-0.5 text-[12px] text-ink-2">
            Net {result.surplusSgp >= 0 ? "+" : ""}
            {result.surplusSgp.toFixed(2)} SGP · {Math.round(result.confidencePct)}% confidence
          </div>
        </div>
        {result.grade && (
          <span
            className={cn(
              "flex size-14 items-center justify-center rounded-2xl font-display text-[30px] font-extrabold leading-none",
              gradeBadgeClass(result.grade),
            )}
          >
            {result.grade}
          </span>
        )}
      </div>

      {result.summary && <p className="mb-3 max-w-[70ch] text-[13px] text-ink-2">{result.summary}</p>}

      {result.categoryImpacts.length > 0 && (
        <div className="mb-1">
          <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wide text-ink-3">
            Category impact (SGP)
          </div>
          <div className="flex flex-wrap gap-1.5">
            {result.categoryImpacts.map((c) => (
              <span
                key={c.cat}
                className={cn(
                  "tnum inline-flex items-center gap-0.5 rounded-md px-2 py-0.5 text-[11px] font-bold",
                  c.delta >= 0 ? "bg-ok/12 text-ok" : "bg-ember/12 text-ember",
                )}
              >
                {c.cat} {c.delta >= 0 ? "+" : ""}
                {c.delta.toFixed(1)}
              </span>
            ))}
          </div>
        </div>
      )}

      {result.deltaPlayoffProb !== undefined && (
        <div className="mt-3 border-t border-line pt-3">
          <span
            className={cn(
              "tnum inline-flex items-center gap-1 text-[12px] font-semibold",
              result.deltaPlayoffProb >= 0 ? "text-ok" : "text-ember",
            )}
          >
            <TrendingUp className="size-3.5" aria-hidden />
            {result.deltaPlayoffProb >= 0 ? "+" : ""}
            {result.deltaPlayoffProb.toFixed(1)}% playoff odds
          </span>
        </div>
      )}

      {result.warnings.length > 0 && (
        <ul className="mt-3 space-y-1">
          {result.warnings.map((w, i) => (
            <li key={i} className="flex items-start gap-1.5 text-[12px] text-warn">
              <TriangleAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden />
              {w}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
