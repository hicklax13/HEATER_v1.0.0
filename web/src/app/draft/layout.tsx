import type { Metadata } from "next";

// Server-component segment layout: its metadata title feeds the root template
// ("HEATER — %s") so the SSR <title> is correct on a hard load / shared-link
// (a client document.title effect can't win against Next's resolved metadata).
export const metadata: Metadata = { title: "Draft" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
