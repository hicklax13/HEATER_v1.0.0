"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import {
  Search,
  ChevronDown,
  Home,
  User,
  Settings,
  Sparkles,
  LogOut,
  Menu,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { usePalette } from "./Providers";
import { HexMesh } from "@/components/ui/HexMesh";

const NAV = [
  { label: "Team", href: "/" },
  { label: "Optimizer", href: "/optimizer" },
  { label: "Matchup", href: "/matchup" },
  { label: "Trades", href: "/trades" },
  { label: "Players", href: "/players" },
  { label: "Research", href: "/research" },
];

export function TopBar() {
  const pathname = usePathname();
  const palette = usePalette();
  const [hovered, setHovered] = useState<string | null>(null);
  const reduce = useReducedMotion();
  const activeLabel = NAV.find((n) => n.href === pathname)?.label ?? "Team";
  const underlineOn = hovered ?? activeLabel;

  return (
    <header className="sticky top-0 z-40 overflow-hidden border-b border-white/[0.06] bg-gradient-to-b from-[#15294a] to-navy shadow-[0_1px_0_rgba(255,92,16,0.5),0_10px_30px_rgba(11,24,48,0.18)]">
      <HexMesh />
      <div className="relative flex h-[68px] w-full items-center gap-4 pl-3 pr-5">
        <Link
          href="/"
          aria-label="HEATER — back to Home"
          className="group flex shrink-0 items-center rounded-xl"
        >
          <Image
            src="/brand/heater-wordmark-v2.png"
            alt="HEATER home"
            width={104}
            height={54}
            priority
            className="h-[48px] w-auto drop-shadow-[0_3px_9px_rgba(0,0,0,0.45)] transition-transform duration-[var(--dur-1)] group-hover:scale-[1.04] motion-reduce:transform-none"
          />
        </Link>

        {/* mobile nav — hamburger dropdown (desktop uses the inline <nav> below) */}
        <DropdownMenu.Root>
          <DropdownMenu.Trigger
            aria-label="Open navigation menu"
            className="flex size-10 items-center justify-center rounded-lg text-white/80 transition-colors hover:bg-white/10 md:hidden"
          >
            <Menu className="size-6" aria-hidden />
          </DropdownMenu.Trigger>
          <DropdownMenu.Portal>
            <DropdownMenu.Content
              align="start"
              sideOffset={10}
              className="z-50 min-w-[210px] rounded-xl border border-white/10 bg-navy p-1.5 text-chrome shadow-[0_20px_60px_rgba(0,0,0,0.5)]"
            >
              {NAV.map((n) => {
                const active = n.label === activeLabel;
                return (
                  <DropdownMenu.Item key={n.label} asChild>
                    <Link
                      href={n.href}
                      aria-current={active ? "page" : undefined}
                      className={cn(
                        "flex min-h-11 cursor-pointer items-center rounded-lg px-3 py-2 text-sm font-semibold outline-none transition-colors data-[highlighted]:bg-white/10",
                        active ? "text-heat" : "text-chrome",
                      )}
                    >
                      {n.label}
                    </Link>
                  </DropdownMenu.Item>
                );
              })}
            </DropdownMenu.Content>
          </DropdownMenu.Portal>
        </DropdownMenu.Root>

        <nav
          className="ml-1 hidden items-center gap-1 md:flex"
          onMouseLeave={() => setHovered(null)}
        >
          <Link
            href="/"
            aria-label="Home"
            className="flex size-11 items-center justify-center rounded-lg text-white/70 transition-colors hover:bg-white/5 hover:text-white"
          >
            <Home className="size-5" aria-hidden />
          </Link>
          {NAV.map((n) => {
            const active = n.label === activeLabel;
            return (
              <Link
                key={n.label}
                href={n.href}
                onMouseEnter={() => setHovered(n.label)}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "relative flex h-11 items-center px-3 text-sm font-semibold transition-colors",
                  active ? "text-white" : "text-white/70 hover:text-white",
                )}
              >
                {n.label}
                {underlineOn === n.label && (
                  <motion.span
                    layoutId="navline"
                    aria-hidden
                    className="absolute inset-x-2 bottom-1.5 h-[2px] rounded-full bg-heat shadow-[0_0_10px_rgba(255,92,16,0.8)]"
                    transition={
                      reduce ? { duration: 0 } : { type: "spring", stiffness: 500, damping: 36 }
                    }
                  />
                )}
              </Link>
            );
          })}
        </nav>

        <div className="flex-1" />

        <button
          onClick={palette.toggle}
          aria-label="Search players and pages (Command or Control K)"
          className="flex h-9 min-h-9 items-center gap-2 rounded-lg border border-line bg-canvas px-3 text-sm font-semibold text-ink-2 transition-colors hover:bg-surface"
        >
          <Search className="size-4 text-ink-3" aria-hidden />
          <span className="hidden lg:inline">Search</span>
          <kbd className="tnum hidden rounded bg-surface-2 px-1.5 py-0.5 text-[11px] font-medium text-ink-3 lg:inline">
            ⌘K
          </kbd>
        </button>

        <span className="tnum rounded-md bg-gradient-to-b from-[#ff7a2e] to-heat px-2.5 py-1 text-[11px] font-bold tracking-wider text-white shadow-[0_2px_8px_rgba(255,92,16,0.35)]">
          PRO
        </span>

        <DropdownMenu.Root>
          <DropdownMenu.Trigger
            aria-label="Account menu"
            className="flex items-center gap-1 rounded-full"
          >
            <span className="flex size-9 items-center justify-center rounded-full bg-gradient-to-b from-[#ff7a2e] to-heat font-display text-[13px] font-bold text-white shadow-[0_2px_8px_rgba(255,92,16,0.35)]">
              CH
            </span>
            <ChevronDown className="size-4 text-white/50" aria-hidden />
          </DropdownMenu.Trigger>
          <DropdownMenu.Portal>
            <DropdownMenu.Content
              align="end"
              sideOffset={8}
              className="z-50 min-w-[208px] rounded-xl border border-white/10 bg-navy p-1.5 text-chrome shadow-[0_20px_60px_rgba(0,0,0,0.5)]"
            >
              <div className="px-2.5 py-2">
                <div className="text-sm font-medium text-white">Connor Hickey</div>
                <div className="text-xs text-white/50">Team Hickey</div>
              </div>
              <DropdownMenu.Separator className="my-1 h-px bg-white/10" />
              <MenuItem icon={User} label="Profile" />
              <MenuItem icon={Settings} label="Settings" />
              <MenuItem icon={Sparkles} label="Manage Pro" accent />
              <DropdownMenu.Separator className="my-1 h-px bg-white/10" />
              <MenuItem icon={LogOut} label="Sign out" />
            </DropdownMenu.Content>
          </DropdownMenu.Portal>
        </DropdownMenu.Root>
      </div>
    </header>
  );
}

function MenuItem({
  icon: Icon,
  label,
  accent,
}: {
  icon: LucideIcon;
  label: string;
  accent?: boolean;
}) {
  return (
    <DropdownMenu.Item
      className={cn(
        "flex min-h-9 cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm outline-none data-[highlighted]:bg-white/10",
        accent ? "text-flame" : "text-chrome",
      )}
    >
      <Icon className="size-4" aria-hidden />
      {label}
    </DropdownMenu.Item>
  );
}
