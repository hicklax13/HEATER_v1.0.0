"use client";

import * as Tooltip from "@radix-ui/react-tooltip";
import { createContext, useContext, useEffect, useState } from "react";
import { CommandPalette } from "./CommandPalette";

type PaletteCtx = { open: boolean; setOpen: (v: boolean) => void; toggle: () => void };
const Ctx = createContext<PaletteCtx | null>(null);

export function usePalette(): PaletteCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("usePalette must be used within <Providers>");
  return c;
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  return (
    <Ctx.Provider value={{ open, setOpen, toggle: () => setOpen((o) => !o) }}>
      <Tooltip.Provider delayDuration={200} skipDelayDuration={300}>
        {children}
        <CommandPalette open={open} onOpenChange={setOpen} />
      </Tooltip.Provider>
    </Ctx.Provider>
  );
}
