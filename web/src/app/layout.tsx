import type { Metadata } from "next";
import { Archivo, Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/chrome/Providers";
import { TopBar } from "@/components/chrome/TopBar";
import { Bubba } from "@/components/bubba/Bubba";
import { BubbaContextProvider } from "@/components/bubba/BubbaContext";
import { AuthProvider } from "@/components/auth/AuthProvider";
import { SubscriptionProvider } from "@/lib/use-subscription";

// Display: Archivo variable, with the width (wdth) axis for stretched hero numerals.
const archivo = Archivo({
  subsets: ["latin"],
  axes: ["wdth"],
  variable: "--font-archivo",
  display: "swap",
});
// Body: Inter variable.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});
export const metadata: Metadata = {
  // `default` titles the root route ("/" = My Team); `template` wraps each route
  // segment's own metadata title (e.g. "Standings" → "HEATER — Standings") so the
  // correct title is in the SSR <title> on a hard load / shared-link, which a
  // client document.title effect can't win against (Next owns <title>).
  title: { default: "HEATER — My Team", template: "HEATER — %s" },
  description: "The analyst's edge for fantasy baseball.",
  icons: {
    icon: "/brand/heater-icon-32.png",
    apple: "/brand/heater-icon-180.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${archivo.variable} ${inter.variable}`}
    >
      <body className="min-h-full bg-canvas text-ink">
        <AuthProvider>
          <SubscriptionProvider>
            <Providers>
              {/* BubbaContextProvider wraps BOTH the page tree ({children}, which
                  publishes its loaded data via usePageData) AND <Bubba/> (which
                  reads it) so the publisher and reader share one context. */}
              <BubbaContextProvider>
                {/* Skip-to-content link: visually hidden until focused (sr-only + focus:not-sr-only). */}
                <a
                  href="#main-content"
                  className="sr-only focus:not-sr-only focus:fixed focus:left-2 focus:top-2 focus:z-[9999] focus:rounded focus:bg-heat focus:px-3 focus:py-2 focus:text-sm focus:font-bold focus:text-white focus:outline-none"
                >
                  Skip to content
                </a>
                <div className="flex min-h-full flex-col">
                  <TopBar />
                  <div id="main-content" className="contents">
                    {children}
                  </div>
                </div>
                <Bubba />
              </BubbaContextProvider>
            </Providers>
          </SubscriptionProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
