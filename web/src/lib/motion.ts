import { animate, useReducedMotion, type Transition, type Variants } from "framer-motion";
import { useEffect, useState } from "react";

/** Crisp snap easing — mirrors --ease-snap. */
export const EASE_SNAP: [number, number, number, number] = [0.2, 0.8, 0.2, 1];

/** Duration tokens (seconds) — mirror --dur-1 / --dur-2. */
export const DUR = { fast: 0.12, mid: 0.22 } as const;

/** Single shared spring for card/button hovers. */
export const SPRING: Transition = {
  type: "spring",
  stiffness: 420,
  damping: 30,
  mass: 0.7,
};

/** Staggered scroll-reveal: small opacity + rise, once, fast. */
export const revealVariants: Variants = {
  hidden: { opacity: 0, y: 10 },
  show: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: DUR.mid, ease: EASE_SNAP, delay: i * 0.05 },
  }),
};

/** Count a number up to `value` on mount; static under reduced-motion.
 *  Returns a DERIVED value when reduced (no synchronous setState in the effect). */
export function useCountUp(value: number, duration = 0.9): number {
  const reduce = useReducedMotion();
  const [n, setN] = useState(0);
  useEffect(() => {
    if (reduce) return;
    const controls = animate(0, value, {
      duration,
      ease: EASE_SNAP,
      onUpdate: (v) => setN(Math.round(v)),
    });
    return () => controls.stop();
  }, [value, duration, reduce]);
  return reduce ? value : n;
}

/** Mount stagger: container orchestrates, item rises in. */
export const staggerContainer: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.04 } },
};
export const staggerItem: Variants = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.32, ease: EASE_SNAP } },
};
