/** Engineered hero numeral — exploits Archivo's width (wdth) axis + tabular figures. */
export function HeroNum({
  children,
  width = 78,
  className,
  style,
}: {
  children: React.ReactNode;
  width?: number; // Archivo wdth axis (62..125)
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <span
      className={className}
      style={{
        fontFamily: "var(--font-display), system-ui, sans-serif",
        fontVariationSettings: `"wdth" ${width}`,
        fontVariantNumeric: "tabular-nums",
        fontWeight: 800,
        letterSpacing: "-0.01em",
        ...style,
      }}
    >
      {children}
    </span>
  );
}
