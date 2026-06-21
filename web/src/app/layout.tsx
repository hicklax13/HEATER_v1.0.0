import type { Metadata } from "next";
import { Archivo, Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/chrome/Providers";
import { TopBar } from "@/components/chrome/TopBar";
import { Bubba } from "@/components/bubba/Bubba";
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
  title: "HEATER — My Team",
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
              <div className="flex min-h-full flex-col">
                <TopBar />
                {children}
              </div>
              <Bubba />
            </Providers>
          </SubscriptionProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
