import type { Metadata } from "next";
import { Archivo, Inter, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/chrome/Providers";

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
// Figures: IBM Plex Mono (static weights).
const plex = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex",
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
      className={`${archivo.variable} ${inter.variable} ${plex.variable}`}
    >
      <body className="min-h-full bg-canvas text-ink">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
