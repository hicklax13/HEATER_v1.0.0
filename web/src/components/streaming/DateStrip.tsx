"use client";

import { formatStreamDate } from "@/lib/streaming-data";

/** today + the next 7 days as canonical YYYY-MM-DD (LOCAL date, no UTC shift). */
export function next7Days(): string[] {
  const out: string[] = [];
  const base = new Date();
  base.setHours(0, 0, 0, 0);
  for (let i = 0; i <= 7; i++) {
    const d = new Date(base);
    d.setDate(base.getDate() + i);
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    out.push(`${yyyy}-${mm}-${dd}`);
  }
  return out;
}

export function DateStrip({
  days,
  selected,
  onSelect,
}: {
  days: string[];
  selected: string;
  onSelect: (d: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2" role="tablist" aria-label="Streaming date">
      {days.map((d, i) => {
        const active = d === selected;
        return (
          <button
            key={d}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onSelect(d)}
            className={
              "rounded-xl border px-3 py-2 text-[13px] font-semibold transition-colors " +
              (active
                ? "border-heat bg-heat text-white"
                : "border-line bg-canvas text-ink-2 hover:border-heat/50 hover:text-navy")
            }
          >
            {i === 0 ? "Today" : formatStreamDate(d)}
          </button>
        );
      })}
    </div>
  );
}
