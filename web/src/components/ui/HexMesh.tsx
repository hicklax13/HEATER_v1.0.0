import { HEX, HEX_SIZE } from "@/lib/hex";

/**
 * Tiled hexagon mesh overlay for dark panels (top bar, heroes).
 * Pass `par` for cursor parallax; omit for a static (ember-drift only) layer.
 */
export function HexMesh({ par }: { par?: { x: number; y: number } }) {
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 motion-safe:animate-[ember-drift_22s_linear_infinite]"
      style={{
        backgroundImage: `url("${HEX}")`,
        backgroundSize: HEX_SIZE,
        transform: par ? `translate(${par.x}px, ${par.y}px)` : undefined,
      }}
    />
  );
}
