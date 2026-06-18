"use client";

import { useEffect, useState } from "react";
import { ArrowRight, ArrowUp, ArrowDown, TrendingUp, Plus } from "lucide-react";
import { fetchTrades, type TradesData, type TradeRec, type TradePlayer, type CatImpact } from "@/lib/trades-data";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { PlayerLink } from "@/components/player/PlayerLink";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { cn } from "@/lib/utils";

const YOU = "Team Hickey";

export default function TradesPage() {
  const [data, setData] = useState<TradesData | null>(null);

  useEffect(() => {
    let alive = true;
    fetchTrades().then((d) => {
      if (alive) setData(d);
    });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {!data ? (
          <LoadingView />
        ) : (
          <div className="space-y-6">
            <Header needs={data.needs} count={data.recs.length} />
            <div className="space-y-4">
              {data.recs.map((rec) => (
                <TradeCard key={rec.id} rec={rec} />
              ))}
            </div>
          </div>
        )}
      </main>
      <Footer freshnessMinutes={9} />
    </>
  );
}

function Header({ needs, count }: { needs: string[]; count: number }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
          Trade Finder · Week 13
        </div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Trades</h1>
        <p className="mt-1 text-[13px] text-ink-2">
          {count} deals that target your needs:{" "}
          {needs.map((n, i) => (
            <span key={n}>
              <span className="font-bold text-heat">{n}</span>
              {i < needs.length - 1 ? ", " : ""}
            </span>
          ))}
          .
        </p>
      </div>
      <button className="inline-flex min-h-10 items-center gap-2 rounded-xl border border-line bg-canvas px-4 text-sm font-bold text-navy transition-colors hover:bg-surface">
        <Plus className="size-4" aria-hidden />
        Build A Trade
      </button>
    </div>
  );
}

function gradeColor(grade: string): string {
  const g = grade[0];
  if (g === "A") return "bg-ok/12 text-ok";
  if (g === "B") return "bg-heat/12 text-heat";
  return "bg-surface-2 text-ink-2";
}

function TradeCard({ rec }: { rec: TradeRec }) {
  return (
    <Card className="p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-[10px] font-bold uppercase tracking-wide text-ink-3">Trade With</div>
          <div className="font-display text-base font-bold text-navy">
            {rec.partner}
            <span className="tnum ml-2 text-[12px] font-medium text-ink-3">{rec.partnerRecord}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[12px] font-semibold text-ink-2">{rec.verdict}</span>
          <span className={cn("rounded-lg px-2.5 py-1 font-display text-base font-extrabold", gradeColor(rec.grade))}>
            {rec.grade}
          </span>
        </div>
      </div>

      <div className="grid items-center gap-3 md:grid-cols-[1fr_auto_1fr]">
        <Side label="You Give" players={rec.give} rosteredBy={YOU} tone="give" />
        <div className="flex justify-center">
          <span className="flex size-9 items-center justify-center rounded-full bg-surface text-ink-3">
            <ArrowRight className="size-5" aria-hidden />
          </span>
        </div>
        <Side label="You Get" players={rec.get} rosteredBy={rec.partner} tone="get" />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-1.5">
        {rec.impact.map((im) => (
          <ImpactChip key={im.cat} im={im} />
        ))}
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
        <p className="max-w-[60ch] text-[12.5px] text-ink-2">{rec.rationale}</p>
        <div className="flex items-center gap-3">
          <span className="tnum inline-flex items-center gap-1 text-[12px] font-semibold text-ok">
            <TrendingUp className="size-3.5" aria-hidden />
            +{rec.playoffDelta}% playoff odds
          </span>
          <button className="inline-flex min-h-9 items-center gap-1 rounded-lg bg-gradient-to-b from-[#ff7a2e] to-heat px-4 text-sm font-bold text-white shadow-[0_4px_12px_rgba(255,92,16,0.3)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none">
            Analyze
          </button>
        </div>
      </div>
    </Card>
  );
}

function Side({
  label,
  players,
  rosteredBy,
  tone,
}: {
  label: string;
  players: TradePlayer[];
  rosteredBy: string;
  tone: "give" | "get";
}) {
  return (
    <div>
      <div
        className={cn(
          "mb-1.5 text-[10px] font-bold uppercase tracking-wide",
          tone === "give" ? "text-ember" : "text-ok",
        )}
      >
        {label}
      </div>
      <div className="space-y-1.5">
        {players.map((p) => (
          <div key={p.name} className="flex items-center gap-2 rounded-lg border border-line bg-surface px-2.5 py-1.5">
            <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={28} />
            <div className="min-w-0">
              <PlayerLink player={{ ...p, rosteredBy }} className="text-[13px]" />
              <div className="tnum text-[10.5px] text-ink-3">{p.keyStat}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ImpactChip({ im }: { im: CatImpact }) {
  const up = im.dir === "up";
  const Icon = up ? ArrowUp : ArrowDown;
  return (
    <span
      className={cn(
        "tnum inline-flex items-center gap-0.5 rounded-md px-2 py-0.5 text-[11px] font-bold",
        up ? "bg-ok/12 text-ok" : "bg-ember/12 text-ember",
      )}
    >
      <Icon className="size-3" aria-hidden />
      {im.cat} {im.delta}
    </span>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-14 w-56" />
      <Skeleton className="h-52 w-full rounded-2xl" />
      <Skeleton className="h-52 w-full rounded-2xl" />
      <Skeleton className="h-52 w-full rounded-2xl" />
    </div>
  );
}
