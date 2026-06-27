"use client";

import { cn } from "@/lib/utils";
import { SCOPE_LABELS, type Scope } from "@/lib/start-sit-data";

const SCOPES: Scope[] = ["today", "rest_of_week", "rest_of_season"];

/** 3-button segmented horizon control. Mirrors the Optimizer's scope tabs. */
export function ScopeSelector({ value, onChange }: { value: Scope; onChange: (s: Scope) => void }) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-surface p-1" role="tablist" aria-label="Horizon">
      {SCOPES.map((s) => (
        <button
          key={s}
          role="tab"
          aria-selected={value === s}
          onClick={() => onChange(s)}
          className={cn(
            "min-h-9 rounded-lg px-3.5 py-1.5 text-[13px] font-bold transition-colors",
            value === s
              ? "bg-gradient-to-b from-heat-bright to-heat text-white shadow-[0_4px_12px_rgba(255,92,16,0.3)]"
              : "text-ink-2 hover:bg-surface-2",
          )}
        >
          {SCOPE_LABELS[s]}
        </button>
      ))}
    </div>
  );
}
