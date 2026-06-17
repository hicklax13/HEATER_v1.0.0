import type { Transition, Variants } from "framer-motion";

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
