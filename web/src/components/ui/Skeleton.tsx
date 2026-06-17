import { cn } from "@/lib/utils";

/** Shimmer placeholder. `animate-pulse` is disabled by the reduced-motion rule. */
export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("animate-pulse rounded-md bg-surface-2", className)} />;
}
