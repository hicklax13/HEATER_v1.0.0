import { cn } from "@/lib/utils";

/** Surface card with a machined top-edge inset (legacy Combustion detail). */
export function Card({
  className,
  machined = true,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & { machined?: boolean }) {
  return (
    <div
      className={cn(
        "relative rounded-2xl border border-line bg-canvas",
        "shadow-[0_1px_2px_rgba(16,32,55,0.05),0_10px_30px_rgba(16,32,55,0.06)]",
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
