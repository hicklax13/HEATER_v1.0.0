import { COLORS } from "@/lib/tokens";

/**
 * Win-% by category radar (hand-rolled SVG, Combustion-themed).
 * Bigger polygon = stronger team; dashed ring marks 50% break-even.
 */
export function CategoryRadar({
  data,
  size = 264,
}: {
  data: { cat: string; you: number }[];
  size?: number;
}) {
  const n = data.length;
  const cx = size / 2;
  const cy = size / 2;
  const R = size / 2 - 34; // room for axis labels
  const ang = (i: number) => (-90 + (i * 360) / n) * (Math.PI / 180);
  const pt = (i: number, r: number): [number, number] => [cx + r * Math.cos(ang(i)), cy + r * Math.sin(ang(i))];
  const poly = (vals: number[]) =>
    vals
      .map((v, i) => {
        const [x, y] = pt(i, (Math.max(0, Math.min(100, v)) / 100) * R);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

  const rings = [25, 50, 75, 100];

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Win percentage by category">
      <defs>
        <radialGradient id="radar-fill" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={COLORS.heat} stopOpacity={0.34} />
          <stop offset="100%" stopColor={COLORS.heat} stopOpacity={0.06} />
        </radialGradient>
      </defs>
      {rings.map((rg) => (
        <polygon key={rg} points={poly(data.map(() => rg))} fill="none" stroke={COLORS.line} strokeWidth={1} />
      ))}
      {data.map((_, i) => {
        const [x, y] = pt(i, R);
        return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke={COLORS.line} strokeWidth={1} />;
      })}
      {/* 50% break-even */}
      <polygon
        points={poly(data.map(() => 50))}
        fill="none"
        stroke={COLORS.steel}
        strokeOpacity={0.5}
        strokeWidth={1}
        strokeDasharray="3 3"
      />
      {/* your shape */}
      <polygon
        points={poly(data.map((d) => d.you))}
        fill="url(#radar-fill)"
        stroke={COLORS.heat}
        strokeWidth={2}
        strokeLinejoin="round"
        style={{ filter: "drop-shadow(0 0 6px rgba(255,92,16,0.35))" }}
      />
      {data.map((d, i) => {
        const [x, y] = pt(i, (Math.max(0, Math.min(100, d.you)) / 100) * R);
        return (
          <circle
            key={`d${i}`}
            cx={x}
            cy={y}
            r={3.2}
            fill={COLORS.heat}
            stroke={COLORS.canvas}
            strokeWidth={1.5}
          />
        );
      })}
      {/* axis labels */}
      {data.map((d, i) => {
        const [x, y] = pt(i, R + 14);
        const c = Math.cos(ang(i));
        const anchor = Math.abs(c) < 0.3 ? "middle" : c > 0 ? "start" : "end";
        return (
          <text
            key={`l${i}`}
            x={x.toFixed(1)}
            y={y.toFixed(1)}
            textAnchor={anchor}
            dominantBaseline="middle"
            fontSize="10.5"
            fontWeight="700"
            fill={COLORS.ink2}
          >
            {d.cat}
          </text>
        );
      })}
    </svg>
  );
}
