"use client";

import { motion, useReducedMotion } from "framer-motion";
import { EASE_SNAP } from "@/lib/motion";

/**
 * Route-level transition. `template.tsx` re-mounts on every navigation (unlike
 * `layout.tsx`), so this gives each page a quick crossfade on entry. Opacity
 * only — no transform — so there's zero layout shift, and it's short enough not
 * to compete with each page's own mount stagger (which plays after data loads).
 */
export default function Template({ children }: { children: React.ReactNode }) {
  const reduce = useReducedMotion();
  if (reduce) return <>{children}</>;
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.25, ease: EASE_SNAP }}
    >
      {children}
    </motion.div>
  );
}
