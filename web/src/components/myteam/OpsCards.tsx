"use client";

import { Check, TriangleAlert, CircleAlert } from "lucide-react";
import type { OpsCard } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { COLORS } from "@/lib/tokens";

const STATUS = {
  ok: { color: COLORS.ok, icon: Check, text: "text-ok" },
  warn: { color: COLORS.warn, icon: TriangleAlert, text: "text-warn" },
  bad: { color: COLORS.ember, icon: CircleAlert, text: "text-ember" },
} as const;

export function OpsCards({ cards }: { cards: OpsCard[] }) {
  return (
    <div className="flex flex-col gap-4">
      {cards.map((c) => (
        <OpsCardItem key={c.key} card={c} />
      ))}
    </div>
  );
}

function OpsCardItem({ card }: { card: OpsCard }) {
  const s = STATUS[card.status];
  const Icon = s.icon;
  const pct = Math.min(100, Math.round((card.value / card.total) * 100));
  return (
    <Card className="p-4">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[11px] font-medium uppercase tracking-wider text-ink-3">
            {card.label}
          </div>
          <div className="mt-1 font-display text-3xl font-extrabold text-navy">
            <span className="tnum">{card.value}</span>
            <span className="tnum ml-0.5 text-sm font-normal text-ink-2">
              {" "}
              / {card.total}
              {card.unit ? ` ${card.unit}` : ""}
            </span>
          </div>
        </div>
        {card.key === "ip" ? (
          <Ring pct={pct} color={s.color} />
        ) : (
          <span
            className="flex size-8 items-center justify-center rounded-full"
            style={{ backgroundColor: `${s.color}22` }}
          >
            <Icon className="size-4" style={{ color: s.color }} aria-hidden />
          </span>
        )}
      </div>
      <div className="mt-2 flex items-center gap-1.5 text-[12px]">
        <Icon className="size-3.5 shrink-0" style={{ color: s.color }} aria-hidden />
        <span className="font-medium text-ink-2">{card.verdict}</span>
      </div>
    </Card>
  );
}

function Ring({ pct, color }: { pct: number; color: string }) {
  const r = 14;
  const c = 2 * Math.PI * r;
  return (
    <svg width="36" height="36" viewBox="0 0 36 36" role="img" aria-label={`${pct}% of target`}>
      <circle cx="18" cy="18" r={r} fill="none" stroke={COLORS.surface2} strokeWidth="4" />
      <circle
        cx="18"
        cy="18"
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="4"
        strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - pct / 100)}
        transform="rotate(-90 18 18)"
      />
    </svg>
  );
}
