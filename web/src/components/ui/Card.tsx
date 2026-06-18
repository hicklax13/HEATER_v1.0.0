import { cn } from "@/lib/utils";

const TONES = {
  flat: "border border-line bg-canvas",
  raised: "border border-line bg-canvas shadow-elev-2",
  inset: "border border-line bg-surface",
} as const;

/** Surface card with a machined top-edge inset (legacy Combustion detail).
 *  Pass `interactive` ONLY when the whole card is clickable — it lifts on hover
 *  (an honest affordance). Static data cards leave it off. */
export function Card({
  className,
  machined = true,
  tone = "raised",
  interactive = false,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & {
  machined?: boolean;
  tone?: keyof typeof TONES;
  interactive?: boolean;
}) {
  return (
    <div
      className={cn(
        "relative rounded-2xl",
        TONES[tone],
        interactive &&
          "cursor-pointer transition-[transform,box-shadow] duration-[var(--dur-2)] hover:-translate-y-0.5 hover:shadow-elev-3 motion-reduce:transform-none motion-reduce:transition-none",
        className,
      )}
      {...props}
    >
      {machined && (
        <span
          aria-hidden
          className="pointer-events-none absolute inset-x-4 top-0 h-px bg-gradient-to-r from-transparent via-white/70 to-transparent"
        />
      )}
      {children}
    </div>
  );
}
