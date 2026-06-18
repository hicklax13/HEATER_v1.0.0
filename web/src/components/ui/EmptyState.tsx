import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Standardized empty / error panel — icon chip + title + optional body + action.
 * Use INSIDE a Card for page-level states, or bare inside a table area for
 * no-results. `tone="error"` switches the chip/icon to ember.
 */
export function EmptyState({
  icon: Icon,
  title,
  body,
  tone = "neutral",
  action,
  className,
}: {
  icon: LucideIcon;
  title: string;
  body?: string;
  tone?: "neutral" | "error";
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center px-6 py-10 text-center", className)}>
      <div
        className={cn(
          "flex size-12 items-center justify-center rounded-full",
          tone === "error" ? "bg-ember/10" : "bg-surface-2",
        )}
      >
        <Icon className={cn("size-5", tone === "error" ? "text-ember" : "text-ink-3")} aria-hidden />
      </div>
      <h3 className="mt-4 font-display text-base font-bold text-navy">{title}</h3>
      {body && <p className="mt-1 max-w-sm text-sm text-ink-2">{body}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}
