"use client";

import { useEffect, useMemo, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import * as Tabs from "@radix-ui/react-tabs";
import { X, Star, Minus, ArrowLeftRight, Layers, Plus, Trophy } from "lucide-react";
import type { PlayerRef } from "@/lib/types";
import { MLB } from "@/lib/tokens";
import { teamBrand } from "@/lib/teams";
import {
  getPlayerDetail,
  fetchPlayerDetail,
  skeletonDetail,
  type HistoryEvent,
  type PlayerDetail,
} from "@/lib/player-detail";
import { isLive } from "@/lib/api/live";
import { cn } from "@/lib/utils";

/**
 * Canonical player card. Wrap ANY player reference anywhere in the app:
 *   <PlayerDialog player={ref}><button>…</button></PlayerDialog>
 * or use <PlayerLink player={ref} /> for an inline clickable name.
 */
export type DialogPlayer = PlayerRef & { ownPct?: number; ownDelta?: number; rosteredBy?: string };

const TABS = [
  { value: "games", label: "Games" },
  { value: "season", label: "Season Stats" },
  { value: "proj", label: "Projections" },
  { value: "history", label: "History" },
];

/* eslint-disable @next/next/no-img-element -- remote MLB CDN headshots/logos */
export function PlayerDialog({ player, children }: { player: DialogPlayer; children: React.ReactNode }) {
  // Fetch the REAL card only when the dialog OPENS (it's mounted for every player
  // link — fetching on mount would fire N requests per page). Until then (and on a
  // live error) we show the skeleton: real identity + empty "—" stats, never
  // fabricated numbers. Off-live, the mock is the intended demo behavior.
  const [open, setOpen] = useState(false);
  const [loaded, setLoaded] = useState<PlayerDetail | null>(null);
  const fallback = useMemo(() => (isLive() ? skeletonDetail(player) : getPlayerDetail(player)), [player]);
  useEffect(() => {
    if (!open || loaded) return;
    let active = true;
    fetchPlayerDetail(player)
      .then((r) => active && setLoaded(r))
      .catch(() => {}); // keep the skeleton/mock fallback on a live error
    return () => {
      active = false;
    };
    // Keyed by the stable mlbId, not the `player` object: a parent re-render
    // that passes a new-but-equal player object must not trigger a refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, loaded, player.mlbId]);
  const d = loaded ?? fallback;
  const tb = teamBrand(d.teamId);

  // Rich team-colored fill: lightened edge → primary → deepened primary (reads on bright AND dark teams).
  const headerBg = `linear-gradient(105deg, color-mix(in srgb, ${tb.primary} 88%, white) 0%, ${tb.primary} 56%, color-mix(in srgb, ${tb.primary} 72%, black) 122%)`;

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild>{children}</Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-navy-deep/60 backdrop-blur-sm" />
        <Dialog.Content
          aria-describedby={undefined}
          className="fixed left-1/2 top-1/2 z-50 flex max-h-[88vh] w-[min(96vw,940px)] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-2xl border border-line bg-canvas shadow-[0_40px_100px_rgba(9,20,42,0.5)]"
        >
          <Dialog.Title className="sr-only">{d.name} player card</Dialog.Title>

          {/* ---- team-colored header with prominent logo fill ---- */}
          <div className="relative shrink-0 overflow-hidden px-6 py-6 text-white" style={{ background: headerBg }}>
            {/* soft glow so the logo reads even on same-color teams (e.g. navy logo on navy) */}
            <span
              aria-hidden
              className="pointer-events-none absolute inset-0"
              style={{ background: "radial-gradient(circle at 82% 50%, rgba(255,255,255,0.20), transparent 46%)" }}
            />
            {/* prominent player team-logo fill — white halo makes dark logos read on dark teams */}
            <img
              src={MLB.teamLogo(d.teamId)}
              alt=""
              aria-hidden
              onError={(e) => ((e.currentTarget as HTMLImageElement).style.display = "none")}
              style={{
                filter:
                  "brightness(1.7) drop-shadow(0 0 2px rgba(255,255,255,0.55)) drop-shadow(0 4px 16px rgba(0,0,0,0.25))",
              }}
              className="pointer-events-none absolute -right-3 top-1/2 size-80 -translate-y-1/2 opacity-[0.5]"
            />
            {/* left scrim keeps white text crisp over the logo + bright team colors */}
            <span
              aria-hidden
              className="pointer-events-none absolute inset-y-0 left-0 w-3/4 bg-gradient-to-r from-black/35 via-black/10 to-transparent"
            />

            <Dialog.Close
              aria-label="Close"
              className="absolute right-3 top-3 z-10 flex size-9 items-center justify-center rounded-lg text-white/80 hover:bg-white/15 hover:text-white"
            >
              <X className="size-5" aria-hidden />
            </Dialog.Close>

            <div className="relative flex flex-wrap items-center gap-5">
              <img
                src={MLB.headshot(d.mlbId)}
                alt={d.name}
                onError={(e) => ((e.currentTarget as HTMLImageElement).style.opacity = "0")}
                className="size-24 rounded-full border-2 border-white/40 bg-white/10 object-cover shadow-lg"
              />
              <div className="min-w-0 flex-1">
                <h2 className="font-display text-[26px] font-extrabold leading-tight drop-shadow-sm">{d.name}</h2>
                <div className="mt-1 text-[13px] font-semibold text-white/90">
                  {d.pos} · {d.teamName} {d.jersey} · {d.bats}
                </div>

                {/* white separator under name/# and above headline stats */}
                <div className="my-3 h-px w-full bg-white/35" />

                <div className="flex flex-wrap gap-x-5 gap-y-1">
                  {d.headline.map((s) => (
                    <div key={s.label}>
                      <span className="tnum font-display text-xl font-extrabold">{s.value}</span>
                      <span className="ml-1 text-[10px] font-semibold uppercase tracking-wide text-white/75">
                        {s.label}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <RosterBox owner={d.rosteredBy} ownPct={d.ownPct} ownDelta={d.ownDelta} />
            </div>
          </div>

          {/* ---- tabs ---- */}
          <Tabs.Root defaultValue="games" className="flex min-h-0 flex-1 flex-col">
            <Tabs.List className="flex shrink-0 gap-1 border-b border-line px-4">
              {TABS.map((t) => (
                <Tabs.Trigger
                  key={t.value}
                  value={t.value}
                  className="relative px-3 py-3 text-sm font-bold text-ink-2 outline-none data-[state=active]:text-navy data-[state=active]:after:absolute data-[state=active]:after:inset-x-2 data-[state=active]:after:-bottom-px data-[state=active]:after:h-0.5 data-[state=active]:after:rounded-full data-[state=active]:after:bg-heat"
                >
                  {t.label}
                </Tabs.Trigger>
              ))}
            </Tabs.List>

            <div className="min-h-0 flex-1 overflow-y-auto p-5">
              <Tabs.Content value="games" className="outline-none">
                <GamesTab columns={d.gameColumns} rows={d.gameLog} />
              </Tabs.Content>
              <Tabs.Content value="season" className="outline-none">
                <SeasonTab d={d} />
              </Tabs.Content>
              <Tabs.Content value="proj" className="outline-none">
                <ProjTab d={d} />
              </Tabs.Content>
              <Tabs.Content value="history" className="outline-none">
                <HistoryTab events={d.history} />
              </Tabs.Content>
            </div>
          </Tabs.Root>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

/** Solid-white roster chip — matches the FantasyPros reference layout (no Drop button). */
function RosterBox({ owner, ownPct, ownDelta }: { owner: string; ownPct: number; ownDelta: number }) {
  const isFA = owner === "Free Agent";
  const down = ownDelta < 0;
  const up = ownDelta > 0;
  const deltaTxt = ownDelta === 0 ? "" : ` (${up ? "+" : ""}${ownDelta}% Last Day)`;
  const bannerCls = down ? "bg-ember/10 text-ember" : up ? "bg-ok/10 text-ok" : "bg-surface text-ink-3";

  return (
    <div className="hidden w-[244px] overflow-hidden rounded-xl bg-white text-ink shadow-[0_8px_24px_rgba(0,0,0,0.22)] sm:block">
      <div className="flex items-center gap-2 p-3">
        {/* fantasy team logo (placeholder → Yahoo team_logos url when wired) */}
        {!isFA && (
          <img
            src="/brand/team-logo-placeholder.svg"
            alt=""
            aria-hidden
            className="size-9 shrink-0 rounded-full ring-1 ring-black/5"
          />
        )}
        <div className="min-w-0 flex-1">
          <div className="text-[10px] font-bold uppercase tracking-wide text-ink-3">
            {isFA ? "Status" : "Rostered By"}
          </div>
          <div className="flex items-center gap-1">
            {!isFA && <Trophy className="size-3.5 shrink-0" style={{ color: "#f0b429" }} aria-hidden />}
            <span className="truncate text-[14px] font-bold text-heat">{owner}</span>
          </div>
        </div>
        <button
          aria-label="Add to watchlist"
          className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-line text-ink-2 hover:bg-surface"
        >
          <Star className="size-4" aria-hidden />
        </button>
      </div>
      <div className={cn("tnum px-3 py-1.5 text-center text-[11px] font-semibold", bannerCls)}>
        {ownPct}% Rostered{deltaTxt}
      </div>
    </div>
  );
}

function TH({ children, left }: { children: React.ReactNode; left?: boolean }) {
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

function GamesTab({ columns, rows }: { columns: string[]; rows: ReturnType<typeof getPlayerDetail>["gameLog"] }) {
  return (
    <>
      <h3 className="mb-2 font-display text-sm font-bold text-navy">2026 Season Game Log</h3>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px]">
          <thead>
            <tr className="border-b border-line">
              {columns.map((c, i) => (
                <TH key={c} left={i < 3}>
                  {c}
                </TH>
              ))}
            </tr>
          </thead>
          <tbody className="tnum text-[13px]">
            {rows.map((g, i) => (
              <tr key={i} className={cn("border-b border-line/60", g.upcoming && "text-ink-3")}>
                <td className="px-2.5 py-2 font-medium text-ink">{g.date}</td>
                <td className="px-2.5 py-2 text-ink-2">{g.opp}</td>
                <td className={cn("px-2.5 py-2 font-semibold", resultColor(g.result, g.upcoming))}>{g.result}</td>
                {g.line.map((cell, j) => (
                  <td key={j} className="px-2.5 py-2 text-right text-ink">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function resultColor(result: string, upcoming: boolean): string {
  if (upcoming) return "text-steel";
  if (result.startsWith("W")) return "text-ok";
  if (result.startsWith("L")) return "text-ember";
  return "text-ink-2";
}

function StatTable({ d }: { d: ReturnType<typeof getPlayerDetail> }) {
  return (
    <table className="w-full min-w-[560px]">
      <thead>
        <tr className="border-b border-line">
          <TH left>Category</TH>
          <TH>Season</TH>
          <TH>L30</TH>
          <TH>L14</TH>
          <TH>L7</TH>
          <TH>Avg</TH>
          <TH>Std</TH>
        </tr>
      </thead>
      <tbody className="tnum text-[13px]">
        {d.stats.map((r) => (
          <tr key={r.cat} className="border-b border-line/60">
            <th scope="row" className="px-2.5 py-2 text-left font-bold text-navy">
              {r.cat}
            </th>
            <td className="px-2.5 py-2 text-right font-semibold text-ink">{r.season}</td>
            <td className="px-2.5 py-2 text-right text-ink-2">{r.l30}</td>
            <td className="px-2.5 py-2 text-right text-ink-2">{r.l14}</td>
            <td className="px-2.5 py-2 text-right text-ink-2">{r.l7}</td>
            <td className="px-2.5 py-2 text-right text-ink">{r.avg}</td>
            <td className="px-2.5 py-2 text-right text-ink-3">{r.std}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SeasonTab({ d }: { d: ReturnType<typeof getPlayerDetail> }) {
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-2">
        {d.ranks.map((r) => (
          <div key={r.label} className="rounded-xl border border-line bg-surface px-3 py-2">
            <div className="text-[10px] font-bold uppercase tracking-wide text-ink-3">{r.label}</div>
            <div className="tnum font-display text-lg font-extrabold text-navy">{r.value}</div>
          </div>
        ))}
      </div>
      <div className="overflow-x-auto">
        <StatTable d={d} />
      </div>
      <div>
        <h3 className="mb-2 font-display text-sm font-bold text-navy">
          Prior Seasons{" "}
          <span className="tnum ml-1 text-[11px] font-medium text-ink-3">
            2025 #{d.prior.y2025Rank} · 2024 #{d.prior.y2024Rank}
          </span>
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[360px]">
            <thead>
              <tr className="border-b border-line">
                <TH left>Category</TH>
                <TH>2025</TH>
                <TH>2024</TH>
              </tr>
            </thead>
            <tbody className="tnum text-[13px]">
              {d.prior.rows.map((r) => (
                <tr key={r.cat} className="border-b border-line/60">
                  <th scope="row" className="px-2.5 py-2 text-left font-bold text-navy">
                    {r.cat}
                  </th>
                  <td className="px-2.5 py-2 text-right text-ink">{r.y2025}</td>
                  <td className="px-2.5 py-2 text-right text-ink-2">{r.y2024}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ProjTab({ d }: { d: ReturnType<typeof getPlayerDetail> }) {
  return (
    <>
      <h3 className="mb-2 font-display text-sm font-bold text-navy">Projections — Totals By Period</h3>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px]">
          <thead>
            <tr className="border-b border-line">
              <TH left>Category</TH>
              <TH>Today</TH>
              <TH>Next 7</TH>
              <TH>Next 14</TH>
              <TH>Next 30</TH>
              <TH>ROS</TH>
              <TH>Avg</TH>
              <TH>Std</TH>
            </tr>
          </thead>
          <tbody className="tnum text-[13px]">
            {d.projections.map((r) => (
              <tr key={r.cat} className="border-b border-line/60">
                <th scope="row" className="px-2.5 py-2 text-left font-bold text-navy">
                  {r.cat}
                </th>
                <td className="px-2.5 py-2 text-right text-ink-2">{r.today}</td>
                <td className="px-2.5 py-2 text-right text-ink">{r.n7}</td>
                <td className="px-2.5 py-2 text-right text-ink">{r.n14}</td>
                <td className="px-2.5 py-2 text-right text-ink">{r.n30}</td>
                <td className="px-2.5 py-2 text-right font-semibold text-heat">{r.ros}</td>
                <td className="px-2.5 py-2 text-right text-ink">{r.avg}</td>
                <td className="px-2.5 py-2 text-right text-ink-3">{r.std}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

const HISTORY_ICON = {
  drafted: Layers,
  traded: ArrowLeftRight,
  added: Plus,
  dropped: Minus,
} as const;

function HistoryTab({ events }: { events: HistoryEvent[] }) {
  return (
    <>
      <h3 className="mb-3 font-display text-sm font-bold text-navy">Player History</h3>
      <ul className="space-y-3">
        {events.map((e, i) => {
          const Icon = HISTORY_ICON[e.kind];
          return (
            <li key={i} className="flex items-start gap-3 border-b border-line/60 pb-3">
              <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-surface text-heat">
                <Icon className="size-4" aria-hidden />
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-ink">{e.text}</div>
                <div className="tnum text-[12px] text-ink-3">{e.date}</div>
              </div>
              <span className="tnum shrink-0 text-[12px] font-semibold text-ink-2">{e.member}</span>
            </li>
          );
        })}
      </ul>
    </>
  );
}
