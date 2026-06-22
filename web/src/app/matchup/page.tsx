"use client";

import { motion } from "framer-motion";
import { Trophy, ChevronLeft, ChevronRight } from "lucide-react";
import {
  fetchMatchup,
  type MatchupData,
  type MatchPlayer,
  type RosterRow,
  type TeamSide,
  type GameState,
  type CatCol,
  type LeagueTeam,
} from "@/lib/matchup-data";
import { Footer } from "@/components/chrome/Footer";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { HexMesh } from "@/components/ui/HexMesh";
import { CategoryBattle } from "@/components/viz/CategoryBattle";
import { PlayerDialog } from "@/components/player/PlayerDialog";
import { cn } from "@/lib/utils";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty, PageNotLinked } from "@/components/ui/PageStates";

export default function MatchupPage() {
  const { state, retry } = usePageData(fetchMatchup);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "unlinked" && <PageNotLinked />}
        {state.status === "empty" && (
          <PageEmpty
            icon={Trophy}
            title="No active matchup"
            body="You're between matchups — check back when the next week opens."
          />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={2} />}
    </>
  );
}

function Loaded({ data }: { data: MatchupData }) {
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <ScoreHeader data={data} />
      </motion.div>
      <motion.div variants={staggerItem}>
        <CategoryBattle cats={data.cats} youScore={data.you.score} oppScore={data.opp.score} />
      </motion.div>
      <motion.div variants={staggerItem}>
        <CatTotals data={data} />
      </motion.div>
      <motion.div variants={staggerItem}>
        <RosterCompare
          title="Hitters"
          columns={data.hitterColumns}
          rows={data.hitters}
          totals={data.hitterTotals}
          you={data.you.name}
          opp={data.opp.name}
        />
      </motion.div>
      <motion.div variants={staggerItem}>
        <RosterCompare
          title="Pitchers"
          columns={data.pitcherColumns}
          rows={data.pitchers}
          totals={data.pitcherTotals}
          you={data.you.name}
          opp={data.opp.name}
        />
      </motion.div>
      {/* League scoreboard — wired to /api/matchup `league[]` (Matchup-C). The
          guard hides it only on an empty/degraded response. */}
      {data.league.length > 0 && (
        <motion.div variants={staggerItem}>
          <LeagueMatchups data={data} />
        </motion.div>
      )}
    </motion.div>
  );
}

/* ---------------- shared bits ---------------- */

function initials(name: string): string {
  return name
    .replace(/[^a-zA-Z ]/g, "")
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

function Avatar({ name, size = 40, onDark }: { name: string; size?: number; onDark?: boolean }) {
  return (
    <span
      className={cn(
        "flex shrink-0 items-center justify-center rounded-full bg-gradient-to-b from-navy-700 to-navy font-display font-bold text-white ring-2",
        onDark ? "ring-white/25" : "ring-line",
      )}
      style={{ width: size, height: size, fontSize: size * 0.34 }}
      aria-hidden
    >
      {initials(name)}
    </span>
  );
}

function stateColor(state: GameState): string {
  if (state === "live") return "text-ok";
  if (state === "sched") return "text-steel";
  return "text-ink-3";
}

/* ---------------- score header ---------------- */

function ScoreHeader({ data }: { data: MatchupData }) {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-b from-[#15294a] to-navy text-chrome shadow-[0_1px_0_rgba(255,92,16,0.4),0_18px_44px_rgba(11,24,48,0.28)]">
      <HexMesh />
      <div className="relative p-5">
        <div className="mb-4 flex flex-wrap items-center gap-2 border-b border-white/10 pb-3">
          <div className="flex items-center gap-1 rounded-full border border-white/15 px-1">
            <button className="flex size-7 items-center justify-center rounded-full text-white/60 hover:bg-white/10" aria-label="Previous week">
              <ChevronLeft className="size-4" aria-hidden />
            </button>
            <span className="px-1 text-[13px] font-bold text-white">Week {data.week}</span>
            <button className="flex size-7 items-center justify-center rounded-full text-white/60 hover:bg-white/10" aria-label="Next week">
              <ChevronRight className="size-4" aria-hidden />
            </button>
          </div>
          <div className="flex flex-wrap items-center gap-1 text-[12.5px]">
            {data.dateTabs.map((t, i) => (
              <span
                key={t}
                className={cn(
                  "rounded-md px-2 py-1 font-semibold",
                  i === 0 ? "bg-heat text-white" : "text-white/65 hover:bg-white/10",
                )}
              >
                {t}
              </span>
            ))}
          </div>
        </div>

        <div className="grid items-center gap-4 sm:grid-cols-[1fr_auto_1fr]">
          <TeamHead team={data.you} trophy align="left" />
          <div className="flex items-center justify-center gap-3 font-display">
            <span className="tnum text-4xl font-extrabold text-white">{data.you.score}</span>
            <span className="text-[13px] font-semibold uppercase tracking-wide text-white/50">vs</span>
            <span className="tnum text-4xl font-extrabold text-white/55">{data.opp.score}</span>
          </div>
          <TeamHead team={data.opp} align="right" />
        </div>
      </div>
    </div>
  );
}

function TeamHead({ team, trophy, align }: { team: TeamSide; trophy?: boolean; align: "left" | "right" }) {
  const right = align === "right";
  return (
    <div className={cn("flex items-center gap-3", right && "flex-row-reverse text-right")}>
      <Avatar name={team.name} size={48} onDark />
      <div>
        <div className={cn("flex items-center gap-1", right && "flex-row-reverse")}>
          {trophy && <Trophy className="size-4" style={{ color: "#f0b429" }} aria-hidden />}
          <span className="font-display text-lg font-bold text-white">{team.name}</span>
        </div>
        <div className="text-[12px] text-white/70">{team.manager}</div>
        <div className="tnum text-[12px] text-white/55">{team.record}</div>
      </div>
    </div>
  );
}

/* ---------------- category totals ---------------- */

function CatTotals({ data }: { data: MatchupData }) {
  return (
    <Card className="overflow-hidden p-0">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[820px] text-[13px]">
          <thead>
            <tr className="border-b border-line bg-surface">
              <th className="px-3 py-2 text-left text-[11px] font-bold uppercase tracking-wide text-ink-3">Team</th>
              {data.cats.map((c) => (
                <th key={c.key} className="px-2 py-2 text-right text-[11px] font-bold uppercase tracking-wide text-ink-3">
                  {c.key}
                </th>
              ))}
              <th className="px-3 py-2 text-right text-[11px] font-bold uppercase tracking-wide text-navy">Score</th>
            </tr>
          </thead>
          <tbody>
            <CatRow team={data.you} cats={data.cats} side="you" />
            <CatRow team={data.opp} cats={data.cats} side="opp" />
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function CatRow({ team, cats, side }: { team: TeamSide; cats: CatCol[]; side: "you" | "opp" }) {
  return (
    <tr className="border-b border-line/60 last:border-0">
      <td className="whitespace-nowrap px-3 py-2.5 font-bold text-navy">{team.name}</td>
      {cats.map((c) => {
        const val = side === "you" ? c.you : c.opp;
        const wins = c.win === side;
        return (
          <td
            key={c.key}
            className={cn("tnum px-2 py-2.5 text-right", wins ? "bg-heat/10 font-bold text-navy" : "text-ink-2")}
          >
            {val}
          </td>
        );
      })}
      <td className="tnum px-3 py-2.5 text-right font-display text-base font-extrabold text-navy">{team.score}</td>
    </tr>
  );
}

/* ---------------- roster compare ---------------- */

function RosterCompare({
  title,
  columns,
  rows,
  totals,
  you,
  opp,
}: {
  title: string;
  columns: string[];
  rows: RosterRow[];
  totals: { you: string[]; opp: string[] };
  you: string;
  opp: string;
}) {
  return (
    <Card className="overflow-hidden p-0">
      <div className="border-b border-line bg-surface px-4 py-2.5">
        <h2 className="font-display text-sm font-bold text-navy">{title}</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[1040px] text-[12.5px]">
          <thead>
            <tr className="border-b border-line text-[10.5px] font-bold uppercase tracking-wide text-ink-3">
              <th className="px-3 py-2 text-left">Player</th>
              {columns.map((c) => (
                <th key={`yl-${c}`} className="px-1.5 py-2 text-right">{c}</th>
              ))}
              <th className="bg-surface px-2 py-2 text-center text-navy">Pos</th>
              <th className="px-3 py-2 text-left">Player</th>
              {columns.map((c) => (
                <th key={`or-${c}`} className="px-1.5 py-2 text-right">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={i}
                className="border-b border-line/60 transition-colors duration-[var(--dur-1)] hover:bg-surface/60"
              >
                <PlayerCell p={row.you} rosteredBy={you} />
                <StatCells p={row.you} />
                <td className="bg-surface px-2 py-2 text-center font-bold text-navy">{row.slot}</td>
                <PlayerCell p={row.opp} rosteredBy={opp} />
                <StatCells p={row.opp} />
              </tr>
            ))}
            <tr className="border-t-2 border-line bg-surface font-bold text-navy">
              <td className="px-3 py-2.5">TOTAL</td>
              {totals.you.map((v, i) => (
                <td key={`yt-${i}`} className="tnum px-1.5 py-2.5 text-right">{v}</td>
              ))}
              <td className="bg-surface-2" />
              <td className="px-3 py-2.5">TOTAL</td>
              {totals.opp.map((v, i) => (
                <td key={`ot-${i}`} className="tnum px-1.5 py-2.5 text-right">{v}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function PlayerCell({ p, rosteredBy }: { p: MatchPlayer | null; rosteredBy: string }) {
  if (!p) {
    return <td className="px-3 py-2 text-[12px] italic text-ink-3">Empty</td>;
  }
  return (
    <td className="p-0 align-top">
      <PlayerDialog
        player={{ name: p.name, pos: p.pos, teamAbbr: p.teamAbbr, teamId: p.teamId, mlbId: p.mlbId, rosteredBy }}
      >
        <button className="flex w-full flex-col items-start gap-0.5 rounded px-3 py-2 text-left align-top transition-colors hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-heat/50">
          <span className="flex flex-wrap items-center gap-x-1">
            <span className="text-[12.5px] font-semibold text-navy">{p.name}</span>
            <span className="tnum text-[10.5px] text-ink-3">
              {p.teamAbbr} · {p.pos}
            </span>
            {p.badge && (
              <span
                className={cn(
                  "rounded px-1 text-[9px] font-bold",
                  p.badge === "IL" ? "bg-ember/12 text-ember" : "bg-warn/15 text-warn",
                )}
              >
                {p.badge}
              </span>
            )}
          </span>
          <span className={cn("text-[10.5px]", stateColor(p.state))}>{p.status}</span>
        </button>
      </PlayerDialog>
    </td>
  );
}

function StatCells({ p }: { p: MatchPlayer | null }) {
  const vals = p ? p.stats : ["-", "-", "-", "-", "-", "-", "-"];
  return (
    <>
      {vals.map((v, i) => (
        <td
          key={i}
          className={cn("tnum px-1.5 py-2 text-right align-top", v === "-" ? "text-ink-3" : "text-ink")}
        >
          {v}
        </td>
      ))}
    </>
  );
}

/* ---------------- league matchups ---------------- */

function LeagueMatchups({ data }: { data: MatchupData }) {
  return (
    <Card className="overflow-hidden p-0">
      <div className="flex items-center justify-between border-b border-line bg-surface px-4 py-2.5">
        <h2 className="font-display text-sm font-bold text-navy">Week {data.week} Matchups</h2>
        <span className="text-[11px] font-semibold uppercase tracking-wide text-ok">In Progress</span>
      </div>
      <ul>
        {data.league.map((m, i) => (
          <li
            key={i}
            className="grid grid-cols-[1fr_auto_1fr] items-center gap-3 border-b border-line/60 px-4 py-3 last:border-0"
          >
            <LeagueTeamCell team={m.a} align="right" trophy={i === 0} winner={m.a.score > m.b.score} />
            <div className="tnum flex items-center gap-2 font-display text-lg font-extrabold">
              <span className={m.a.score > m.b.score ? "text-navy" : "text-ink-3"}>{m.a.score}</span>
              <span className="text-[11px] font-semibold text-ink-3">–</span>
              <span className={m.b.score > m.a.score ? "text-navy" : "text-ink-3"}>{m.b.score}</span>
            </div>
            <LeagueTeamCell team={m.b} align="left" winner={m.b.score > m.a.score} />
          </li>
        ))}
      </ul>
    </Card>
  );
}

function LeagueTeamCell({
  team,
  align,
  trophy,
  winner,
}: {
  team: LeagueTeam;
  align: "left" | "right";
  trophy?: boolean;
  winner?: boolean;
}) {
  const right = align === "right";
  return (
    <div className={cn("flex min-w-0 items-center gap-2.5", right ? "flex-row-reverse text-right" : "text-left")}>
      <Avatar name={team.name} size={34} />
      <div className="min-w-0">
        <div className={cn("flex items-center gap-1", right && "flex-row-reverse")}>
          {trophy && <Trophy className="size-3.5 shrink-0" style={{ color: "#f0b429" }} aria-hidden />}
          <span className={cn("truncate font-bold", winner ? "text-navy" : "text-ink")}>{team.name}</span>
        </div>
        <div className="tnum truncate text-[11px] text-ink-3">
          {team.manager} · {team.record}
        </div>
      </div>
    </div>
  );
}

function LoadingView() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-32 w-full rounded-2xl" />
      <Skeleton className="h-20 w-full rounded-2xl" />
      <Skeleton className="h-96 w-full rounded-2xl" />
      <Skeleton className="h-96 w-full rounded-2xl" />
    </div>
  );
}
