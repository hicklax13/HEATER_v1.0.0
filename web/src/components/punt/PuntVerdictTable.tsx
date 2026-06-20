"use client";

import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";
import type { PuntData, PuntVerdict } from "@/lib/punt-data";

const VERDICT: Record<PuntVerdict, { label: string; cls: string; order: number }> = {
  compete: { label: "Compete", cls: "bg-ok/12 text-ok", order: 0 },
  tossup: { label: "Toss-up", cls: "bg-warn/15 text-warn", order: 1 },
  punt: { label: "Punt", cls: "bg-ember/12 text-ember", order: 2 },
};

function Th({ children, left }: { children: React.ReactNode; left?: boolean }) {
  return (
    <th
      scope="col"
      className={cn(
        "whitespace-nowrap px-2.5 py-2 text-[11px] font-bold uppercase tracking-wide text-navy",
        left ? "text-left" : "text-center",
      )}
    >
      {children}
    </th>
  );
}

/** Per-category compete/toss-up/punt verdict, grouped verdict-first. */
export function PuntVerdictTable({ data }: { data: PuntData }) {
  const cats = [...data.cats].sort(
    (a, b) => VERDICT[a.verdict].order - VERDICT[b.verdict].order || a.rank - b.rank,
  );
  return (
    <Card className="overflow-hidden p-0">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[620px] text-[13px]">
          <thead>
            <tr className="border-b border-line">
              <Th left>Category</Th>
              <Th>Rank</Th>
              <Th>Verdict</Th>
              <Th left>Recommendation</Th>
            </tr>
          </thead>
          <tbody>
            {cats.map((c) => (
              <tr key={c.cat} className="border-b border-line/60 transition-colors hover:bg-surface">
                <td className="px-2.5 py-2.5 font-display text-[14px] font-extrabold text-navy">{c.cat}</td>
                <td className="tnum px-2.5 py-2.5 text-center text-ink-2">
                  {c.rank}/{data.nTeams}
                </td>
                <td className="px-2.5 py-2.5 text-center">
                  <span className={cn("rounded px-2 py-0.5 text-[11px] font-bold", VERDICT[c.verdict].cls)}>
                    {VERDICT[c.verdict].label}
                  </span>
                </td>
                <td className="px-2.5 py-2.5 text-[12px] text-ink-2">{c.recommendation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
