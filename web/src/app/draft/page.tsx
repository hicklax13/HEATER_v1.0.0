"use client";

import { motion } from "framer-motion";
import { Loader2, AlertTriangle } from "lucide-react";
import { useDraft } from "@/lib/use-draft";
import type { DraftRec, DraftClock } from "@/lib/draft-data";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { Card } from "@/components/ui/Card";
import { PlayerAvatar } from "@/components/ui/PlayerAvatar";
import { PlayerLink } from "@/components/player/PlayerLink";
import { cn } from "@/lib/utils";
import { heatColor } from "@/lib/tokens";
import { SetupForm } from "@/components/draft/SetupForm";
import { RecCard } from "@/components/draft/RecCard";
import { RosterRail } from "@/components/draft/RosterRail";
import { PageLocked } from "@/components/ui/PageStates";

type Draft = ReturnType<typeof useDraft>;

export default function DraftPage() {
  const d = useDraft();
  return (
    <main className="w-full flex-1 px-5 py-6">
      {d.phase === "setup" && <SetupForm onStart={d.start} busy={d.busy} />}
      {d.phase === "drafting" && <Drafting d={d} />}
      {d.phase === "complete" && <Complete d={d} />}
      {d.phase === "locked" && <PageLocked feature="The Draft Simulator" />}
      {d.phase === "error" && <DraftError d={d} />}
    </main>
  );
}

function DraftError({ d }: { d: Draft }) {
  return (
    <Card className="mx-auto max-w-md p-10 text-center">
      <AlertTriangle className="mx-auto mb-3 size-8 text-heat" aria-hidden />
      <p className="text-[15px] font-bold text-navy">Draft simulation failed</p>
      <p className="mt-1 text-[13px] text-ink-2">
        {d.errorMsg || "An unexpected error occurred. Please try again."}
      </p>
      <div className="mt-5 flex justify-center gap-3">
        <button
          onClick={() => d.config && d.start(d.config)}
          disabled={!d.config}
          className="inline-flex min-h-10 items-center rounded-xl bg-gradient-to-b from-heat-bright to-heat px-5 text-sm font-bold text-white shadow-[0_4px_12px_rgba(255,92,16,0.32)] transition-transform hover:scale-[1.02] active:scale-95 disabled:opacity-50 motion-reduce:transform-none"
        >
          Retry
        </button>
        <button
          onClick={d.reset}
          className="inline-flex min-h-10 items-center rounded-xl border border-line px-5 text-sm font-semibold text-navy transition-colors hover:bg-surface"
        >
          Back to Setup
        </button>
      </div>
    </Card>
  );
}

function Drafting({ d }: { d: Draft }) {
  const { clock, recs, busy, config } = d;
  const featured = recs.slice(0, 3);
  const more = recs.slice(3);
  const yourTurn = !!clock?.isUserTurn && !busy;
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <Header clock={clock} busy={busy} />
      </motion.div>
      <motion.div variants={staggerItem} className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <div className="space-y-5">
          {busy ? (
            <BusyBoard />
          ) : yourTurn && recs.length > 0 ? (
            <>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {featured.map((r) => (
                  <RecCard key={r.player.id} rec={r} onDraft={() => d.pick(r.player)} disabled={busy} />
                ))}
              </div>
              {more.length > 0 && <MoreBoard recs={more} onDraft={(r) => d.pick(r.player)} disabled={busy} />}
            </>
          ) : (
            <EmptyBoard />
          )}
        </div>
        {config && (
          <RosterRail
            myRoster={d.myRoster}
            pickLog={d.pickLog}
            userTeamIndex={config.userTeamIndex}
            onReset={d.reset}
          />
        )}
      </motion.div>
    </motion.div>
  );
}

function Header({ clock, busy }: { clock: DraftClock | null; busy: boolean }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
          Round {clock?.round ?? 1} · Pick {clock?.pickInRound ?? 1}
        </div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Draft Simulator</h1>
      </div>
      <div
        className={cn(
          "inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-[12px] font-bold",
          busy ? "bg-surface-2 text-ink-2" : "bg-heat/12 text-heat",
        )}
      >
        {busy ? (
          <>
            <Loader2 className="size-3.5 animate-spin" aria-hidden /> AI teams drafting…
          </>
        ) : (
          "● Your Pick"
        )}
      </div>
    </div>
  );
}

function BusyBoard() {
  return (
    <Card className="flex flex-col items-center justify-center gap-3 p-12 text-center">
      <Loader2 className="size-8 animate-spin text-heat" aria-hidden />
      <p className="text-[14px] font-semibold text-navy">The AI teams are on the clock…</p>
      <p className="text-[12px] text-ink-3">Advancing to your next pick.</p>
    </Card>
  );
}

function EmptyBoard() {
  return (
    <Card className="p-8 text-center">
      <p className="text-[14px] font-semibold text-navy">Recommendations unavailable</p>
      <p className="mt-1 text-[12px] text-ink-3">
        The engine returned no players for this pick. Reset the draft to try again.
      </p>
    </Card>
  );
}

function MoreBoard({
  recs,
  onDraft,
  disabled,
}: {
  recs: DraftRec[];
  onDraft: (r: DraftRec) => void;
  disabled: boolean;
}) {
  return (
    <Card className="p-4">
      <div className="mb-2 text-[12px] font-bold uppercase tracking-wider text-navy">More Available</div>
      <ul className="divide-y divide-line">
        {recs.map((r) => (
          <li key={r.player.id} className="flex items-center gap-3 py-2">
            <PlayerAvatar mlbId={r.player.mlbId} teamId={r.player.teamId} name={r.player.name} size={30} />
            <div className="min-w-0 flex-1">
              <PlayerLink player={r.player} className="block truncate text-[13px]" />
              <div className="text-[10px] text-ink-3">
                {r.player.pos} · {r.player.teamAbbr}
              </div>
            </div>
            <span className="tnum text-[12px] font-bold" style={{ color: heatColor(r.score) }}>
              {r.score}
            </span>
            <button
              onClick={() => onDraft(r)}
              disabled={disabled}
              className="shrink-0 rounded-lg border border-line px-3 py-1.5 text-[12px] font-bold text-navy transition-colors hover:bg-surface disabled:opacity-50"
            >
              Draft
            </button>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function Complete({ d }: { d: Draft }) {
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem} className="text-center">
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">Draft Complete</div>
        <h1 className="font-display text-3xl font-extrabold text-navy">Your Roster</h1>
        <p className="mt-1 text-[13px] text-ink-2">
          {d.myRoster.length} players drafted across {d.config?.numRounds} rounds.
        </p>
      </motion.div>
      <motion.div variants={staggerItem}>
        <Card className="p-5">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {d.myRoster.map((p, i) => (
              <div key={`${p.id}-${i}`} className="flex items-center gap-2.5 rounded-lg bg-surface px-3 py-2">
                <span className="tnum w-5 shrink-0 text-[11px] font-bold text-ink-3">{i + 1}</span>
                <PlayerAvatar mlbId={p.mlbId} teamId={p.teamId} name={p.name} size={30} />
                <PlayerLink player={p} className="min-w-0 flex-1 truncate text-[13px]" />
                <span className="shrink-0 text-[10px] font-bold uppercase text-ink-3">{p.pos}</span>
              </div>
            ))}
          </div>
        </Card>
      </motion.div>
      <motion.div variants={staggerItem} className="text-center">
        <button
          onClick={d.reset}
          className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-gradient-to-b from-heat-bright to-heat px-6 py-2.5 text-sm font-bold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform hover:scale-[1.02] active:scale-95 motion-reduce:transform-none"
        >
          Start New Draft
        </button>
      </motion.div>
    </motion.div>
  );
}
