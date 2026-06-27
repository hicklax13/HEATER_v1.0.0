import { cn, humanizeAge } from "@/lib/utils";

/** Trust block + live data-freshness pulse (breathing orange dot; amber > 24h).
 *  `freshnessMinutes` undefined → the page has no freshness source: keep the trust
 *  block + a steady pulse but omit the (otherwise fabricated) timestamp. */
export function Footer({ freshnessMinutes }: { freshnessMinutes?: number }) {
  const known = freshnessMinutes !== undefined;
  const { label, stale } = known ? humanizeAge(freshnessMinutes) : { label: "", stale: false };
  return (
    <footer className="mt-10 flex w-full flex-wrap items-center justify-between gap-3 border-t border-line px-5 py-5 text-[12px] text-ink-2">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <span className="text-ink-3">Powered by</span>
        <span className="font-medium text-ink">MLB Stats API</span>
        <span className="text-line">·</span>
        <span className="font-medium text-ink">FanGraphs</span>
        <span className="text-line">·</span>
        <span className="font-medium text-ink">Yahoo</span>
      </div>
      <div className="tnum flex items-center gap-2">
        <span className="relative inline-flex size-2">
          <span
            className={cn(
              "absolute inset-0 rounded-full opacity-70 motion-safe:animate-ping",
              stale ? "bg-tier-3" : "bg-heat",
            )}
          />
          <span
            className={cn("relative inline-flex size-2 rounded-full", stale ? "bg-tier-3" : "bg-heat")}
          />
        </span>
        {known && (
          <span className={stale ? "font-medium text-ink" : "text-ink-2"}>
            {stale ? "Data is stale — updated " : "Updated "}
            {label}
          </span>
        )}
        <span className="text-ink-3">{known ? "· " : ""}HEATER v2.0</span>
      </div>
    </footer>
  );
}
