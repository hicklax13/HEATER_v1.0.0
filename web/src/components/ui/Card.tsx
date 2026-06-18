import { cn } from "@/lib/utils";

const TONES = {
  flat: "border border-line bg-canvas",
  raised: "border border-line bg-canvas shadow-elev-2",
  inset: "border border-line bg-surface",
} as const;

/** Surface card with a machined top-edge inset (legacy Combustion detail). */
export function Card({
  className,
  machined = true,
  tone = "raised",
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & { machined?: boolean; tone?: keyof typeof TONES }) {
  return (
    <div className={cn("relative rounded-2xl", TONES[tone], className)} {...props}>
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
