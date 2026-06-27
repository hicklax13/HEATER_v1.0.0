"use client";

import { createContext, useCallback, useContext, useMemo, useState } from "react";

/** What Bubba "sees": the current route id + the data that page last loaded. */
export interface BubbaPageContext {
  pageId: string;
  data: unknown;
}

interface BubbaContextValue {
  pageId: string;
  data: unknown;
  publish: (pageId: string, data: unknown) => void;
}

const Ctx = createContext<BubbaContextValue>({
  pageId: "",
  data: null,
  // Default no-op publish: if a consumer renders outside the provider (SSR /
  // tests), publishing is a harmless no-op rather than a throw.
  publish: () => {},
});

export function BubbaContextProvider({ children }: { children: React.ReactNode }) {
  const [pageCtx, setPageCtx] = useState<BubbaPageContext>({ pageId: "", data: null });
  const publish = useCallback((pageId: string, data: unknown) => {
    setPageCtx({ pageId, data });
  }, []);
  const value = useMemo(
    () => ({ pageId: pageCtx.pageId, data: pageCtx.data, publish }),
    [pageCtx, publish],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useBubbaContext(): BubbaContextValue {
  return useContext(Ctx);
}
