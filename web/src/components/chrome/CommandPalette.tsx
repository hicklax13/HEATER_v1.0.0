"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { Command } from "cmdk";
import { useRouter } from "next/navigation";
import {
  Home,
  ClipboardList,
  ArrowLeftRight,
  UserSearch,
  Trophy,
  FlaskConical,
  Search,
  User,
} from "lucide-react";

const PAGES = [
  { label: "Team", path: "/", icon: Home },
  { label: "Optimizer", path: "#", icon: ClipboardList },
  { label: "Matchup", path: "#", icon: Trophy },
  { label: "Trades", path: "#", icon: ArrowLeftRight },
  { label: "Players", path: "#", icon: UserSearch },
  { label: "Research", path: "#", icon: FlaskConical },
];

const PLAYERS = [
  "Aaron Judge",
  "Bobby Witt Jr.",
  "Tarik Skubal",
  "Juan Soto",
  "Corbin Carroll",
  "Elly De La Cruz",
];

export function CommandPalette({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
}) {
  const router = useRouter();
  const go = (path: string) => {
    onOpenChange(false);
    if (path !== "#") router.push(path);
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-navy-deep/55 backdrop-blur-sm" />
        <Dialog.Content
          aria-label="Command palette"
          className="fixed left-1/2 top-[16%] z-50 w-[min(92vw,640px)] -translate-x-1/2 overflow-hidden rounded-2xl border border-white/10 bg-navy text-chrome shadow-[0_30px_80px_rgba(0,0,0,0.5)]"
        >
          <Dialog.Title className="sr-only">Command palette</Dialog.Title>
          <Dialog.Description className="sr-only">
            Search players or jump between pages.
          </Dialog.Description>
          <Command label="Command palette" className="font-body">
            <div className="flex items-center gap-2 border-b border-white/10 px-4">
              <Search className="size-4 shrink-0 text-white/50" aria-hidden />
              <Command.Input
                autoFocus
                placeholder="Search players or jump to a page…"
                className="h-12 w-full bg-transparent text-[15px] text-chrome outline-none placeholder:text-white/40"
              />
              <kbd className="tnum rounded bg-white/10 px-1.5 py-0.5 text-[11px] text-white/60">
                Esc
              </kbd>
            </div>
            <Command.List className="max-h-[340px] overflow-y-auto p-2 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[11px] [&_[cmdk-group-heading]]:font-medium [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-white/40">
              <Command.Empty className="px-3 py-8 text-center text-sm text-white/50">
                No results found.
              </Command.Empty>
              <Command.Group heading="Pages">
                {PAGES.map((p) => (
                  <Command.Item
                    key={p.label}
                    value={`page ${p.label}`}
                    onSelect={() => go(p.path)}
                    className="flex cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-sm text-chrome data-[selected=true]:bg-white/10"
                  >
                    <p.icon className="size-4 text-heat" aria-hidden />
                    {p.label}
                  </Command.Item>
                ))}
              </Command.Group>
              <Command.Group heading="Players">
                {PLAYERS.map((n) => (
                  <Command.Item
                    key={n}
                    value={`player ${n}`}
                    onSelect={() => go("#")}
                    className="flex cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-sm text-chrome data-[selected=true]:bg-white/10"
                  >
                    <User className="size-4 text-white/50" aria-hidden />
                    {n}
                  </Command.Item>
                ))}
              </Command.Group>
            </Command.List>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
