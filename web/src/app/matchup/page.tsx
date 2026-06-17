"use client";

import { useEffect, useState } from "react";
import { Swords, Target } from "lucide-react";
import {
  fetchMatchup,
  catStatus,
  type MatchupData,
  type CatMatchup,
  type CatStatus,
} from "@/lib/matchup-data";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { COLORS, heatColor } from "@/lib/tokens";
import { cn } from "@/lib/utils";

/* eslint-disable @next/next/no-img-element -- local SVG team crests */

const STATUS: Record<CatStatus, { label: string; text: string; bg: string; bar: string }> = {
  win: { label: "Winning", text: "text-ok", bg: "bg-ok/12", bar: COLORS.ok },
  loss: { label: "Losing", text: "text-ember", bg: "bg-ember/12", bar: COLORS.ember },
  tossup: { label: "Toss-Up", text: "text-warn", bg: "bg-warn/15", bar: COLORS.warn },
};

export default function MatchupPage() {
  const [data, setData] = useState<MatchupData | null>(null);

  useEffect(() => {
    let alive = true;
    fetchMatchup().then((d) => {
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
            <Header data={data} />
            <Hero data={data} />
            <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
              <CategoryBoard categories={data.categories} />
              <aside className="space-y-4">
                <TargetCard categories={data.categories} />
              </aside>
            </div>
          </div>
        )}
      </main>
      <Footer freshnessMinutes={9} />
    </>
  );
}

function tally(cats: CatMatchup[]) {
  let win = 0,
    loss = 0,
    tossup = 0;
  for (const c of cats) {
    const s = catStatus(c.winPct);
    if (s === "win") win++;
    else if (s === "loss") loss++;
    else tossup++;
  }
  return { win, loss, tossup };
}

function Header({ data }: { data: MatchupData }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
          Matchup · Week {data.week}
        </div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Matchup</h1>
        <p className="mt-1 text-[13px] text-ink-2">vs {data.opp.name}</p>
      </div>
      <span className="tnum rounded-md bg-surface px-2.5 py-1 text-[12px] font-semibold text-ink-2 ring-1 ring-line">
        {data.daysLeft} days left
      </span>
    </div>
  );
}

function Hero({ data }: { data: MatchupData }) {
  const t = tally(data.categories);
  const col = heatColor(data.winPct);
  return (
    <section
      className="relative overflow-hidden rounded-2xl border border-white/10 p-6 text-chrome shadow-[0_24px_60px_rgba(9,20,42,0.35)]"
      style={{ background: `radial-gradient(130% 150% at 50% -20%, ${COLORS.navy700}, ${COLORS.navyDeep} 70%)` }}
      aria-label={`Matchup win probability ${data.winPct} percent. Projected ${t.win} winning, ${t.tossup} toss-up, ${t.loss} losing.`}
    >
      <div className="grid items-center gap-4 md:grid-cols-[1fr_auto_1fr]">
        <TeamSide name={data.you.name} record={data.you.record} logo={data.you.logo} />
        <div className="flex flex-col items-center">
          <div className="font-display text-5xl font-extrabold leading-none" style={{ color: col }}>
            <span className="tnum">{data.winPct}</span>
            <span className="text-2xl">%</span>
          </div>
          <div className="mt-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/70">
            Win Probability
          </div>
          <div className="tnum mt-2 flex items-center gap-2 text-[12px] font-semibold">
            <span className="text-ok">{t.win} W</span>
            <span className="text-warn">{t.tossup} Toss-Up</span>
            <span className="text-ember">{t.loss} L</span>
          </div>
        </div>
        <TeamSide name={data.opp.name} record={data.opp.record} logo={data.opp.logo} right />
      </div>

      {/* 12-category battle strip */}
      <div className="relative mt-5 flex gap-1">
        {data.categories.map((c) => {
          const s = STATUS[catStatus(c.winPct)];
          return (
            <div
              key={c.key}
              className="flex-1 rounded-md py-1 text-center"
              style={{ backgroundColor: `${s.bar}2e` }}
              title={`${c.key}: ${c.winPct}% win`}
            >
              <div className="tnum text-[11px] font-bold" style={{ color: s.bar }}>
                {c.key}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function TeamSide({
  name,
  record,
  logo,
  right,
}: {
  name: string;
  record: string;
  logo: string;
  right?: boolean;
}) {
  return (
    <div className={cn("flex flex-col", right ? "items-end text-right" : "items-start")}>
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-white/65">
        {right ? "Opponent" : "You"}
      </div>
      <img
        src={logo}
        alt=""
        aria-hidden
        className="mt-2 size-12 rounded-full bg-white/5 shadow-[0_4px_14px_rgba(0,0,0,0.35)] ring-2 ring-white/20"
      />
      <div className="mt-2 font-display text-lg font-bold leading-tight text-chrome md:text-xl">{name}</div>
      <div className="tnum mt-0.5 text-[13px] text-white/70">{record}</div>
    </div>
  );
}

function CategoryBoard({ categories }: { categories: CatMatchup[] }) {
  const hitting = categories.filter((c) => c.group === "Hitting");
  const pitching = categories.filter((c) => c.group === "Pitching");
  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center gap-2">
        <Swords className="size-4 text-heat" aria-hidden />
        <h2 className="font-display text-base font-bold text-navy">Category Breakdown</h2>
      </div>
      <CatTable title="Hitting" rows={hitting} />
      <div className="mt-5">
        <CatTable title="Pitching" rows={pitching} />
      </div>
      <p className="mt-3 text-[11px] text-ink-3">L, ERA, WHIP — lower is better.</p>
    </Card>
  );
}

function CatTable({ title, rows }: { title: string; rows: CatMatchup[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px]">
        <thead>
          <tr className="border-b border-line">
            <Th left>{title}</Th>
            <Th>You</Th>
            <Th>Opp</Th>
            <Th left>Win Probability</Th>
            <Th>Status</Th>
          </tr>
        </thead>
        <tbody className="text-[13px]">
          {rows.map((c) => {
            const s = STATUS[catStatus(c.winPct)];
            const youBetter = c.winPct >= 50;
            return (
              <tr key={c.key} className="border-b border-line/60">
                <td className="px-2.5 py-2.5">
                  <span className="font-bold text-navy">{c.key}</span>
                  {c.inverse && <span className="ml-1 text-[10px] text-ink-3">▼</span>}
                </td>
                <td className={cn("tnum px-2.5 py-2.5 text-right", youBetter ? "font-bold text-ink" : "text-ink-2")}>
                  {c.you}
                </td>
                <td className={cn("tnum px-2.5 py-2.5 text-right", !youBetter ? "font-bold text-ink" : "text-ink-2")}>
                  {c.opp}
                </td>
                <td className="px-2.5 py-2.5">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                      <span
                        className="block h-full rounded-full"
                        style={{ width: `${c.winPct}%`, backgroundColor: s.bar }}
                      />
                    </div>
                    <span className="tnum w-8 text-right text-[12px] font-semibold" style={{ color: s.bar }}>
                      {c.winPct}%
                    </span>
                  </div>
                </td>
                <td className="px-2.5 py-2.5 text-right">
                  <span className={cn("inline-flex rounded-md px-2 py-0.5 text-[11px] font-bold", s.bg, s.text)}>
                    {s.label}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Th({ children, left }: { children: React.ReactNode; left?: boolean }) {
  return (
    <th
      scope="col"
      className={cn(
        "whitespace-nowrap px-2.5 py-2 text-[11px] font-bold uppercase tracking-wide text-navy",
        left ? "text-left" : "text-right",
      )}
    >
      {children}
    </th>
  );
}

function TargetCard({ categories }: { categories: CatMatchup[] }) {
  const tossups = categories
    .filter((c) => catStatus(c.winPct) === "tossup")
    .sort((a, b) => Math.abs(50 - a.winPct) - Math.abs(50 - b.winPct));
  return (
    <Card className="p-4">
      <div className="mb-2 flex items-center gap-2 text-[12px] font-bold uppercase tracking-wider text-navy">
        <Target className="size-4 text-heat" aria-hidden />
        Toss-Ups To Target
      </div>
      {tossups.length === 0 ? (
        <p className="text-[13px] text-ink-2">No coin-flip categories this week.</p>
      ) : (
        <ul className="space-y-2">
          {tossups.map((c) => (
            <li key={c.key} className="flex items-center justify-between rounded-lg bg-surface px-3 py-2">
              <span className="font-bold text-navy">{c.key}</span>
              <span className="tnum text-[12px] text-ink-2">
                {c.you} <span className="text-ink-3">vs</span> {c.opp}
              </span>
              <span className="tnum text-[12px] font-bold text-warn">{c.winPct}%</span>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-[11px] text-ink-3">
        These are within reach — winning them flips the matchup.
      </p>
    </Card>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-14 w-64" />
      <Skeleton className="h-44 w-full rounded-2xl" />
      <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <Skeleton className="h-96 w-full rounded-2xl" />
        <Skeleton className="h-60 w-full rounded-2xl" />
      </div>
    </div>
  );
}
