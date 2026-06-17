"use client";

import { PlayerDialog, type DialogPlayer } from "./PlayerDialog";
import { cn } from "@/lib/utils";

/**
 * Inline clickable player name → opens the full PlayerDialog.
 * Use ANYWHERE a player's name appears (tables, rosters, lists, search).
 *   <PlayerLink player={ref} />            // renders the name
 *   <PlayerLink player={ref}>Custom</PlayerLink>
 * To make a whole row/card clickable instead, wrap it directly in <PlayerDialog>.
 */
export function PlayerLink({
  player,
  className,
  children,
}: {
  player: DialogPlayer;
  className?: string;
  children?: React.ReactNode;
}) {
  return (
    <PlayerDialog player={player}>
      <button
        type="button"
        className={cn(
          "rounded text-left font-semibold text-navy underline-offset-2 transition-colors hover:text-heat hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-heat/50",
          className,
        )}
      >
        {children ?? player.name}
      </button>
    </PlayerDialog>
  );
}
