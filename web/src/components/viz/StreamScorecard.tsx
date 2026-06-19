"use client";

import { HeatGauge } from "@/components/viz/HeatGauge";
import { FactorBars } from "@/components/viz/FactorBars";
import type { StreamComponents } from "@/lib/streaming-data";

/** The Pitcher Streaming signature instrument: a 0–100 Stream Score dial
 *  (reusing HeatGauge) over the 6 diverging factor bars. Used on the Analyze
 *  panel and the board "why". */
export function StreamScorecard({
  score,
  components,
  size = 168,
}: {
  score: number;
  components: StreamComponents;
  size?: number;
}) {
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="rounded-2xl bg-navy px-4 pb-3 pt-1">
        <HeatGauge value={score} label="Stream Score" size={size} />
      </div>
      <div className="w-full">
        <FactorBars components={components} />
      </div>
    </div>
  );
}
