"use client";

import { useState } from "react";
import { MLB } from "@/lib/tokens";
import { cn } from "@/lib/utils";

/* eslint-disable @next/next/no-img-element -- remote MLB CDN headshots; next/image
   would require remotePatterns config and per-image sizing we don't need here. */
export function PlayerAvatar({
  mlbId,
  teamId,
  name,
  size = 54,
  ring = "ring-line",
  className,
}: {
  mlbId: number;
  teamId: number;
  name: string;
  size?: number;
  ring?: string;
  className?: string;
}) {
  const [imgOk, setImgOk] = useState(true);
  const [logoOk, setLogoOk] = useState(true);
  const initials = name
    .split(" ")
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
  const badge = Math.round(size * 0.42);

  return (
    <span className={cn("relative inline-block", className)} style={{ width: size, height: size }}>
      <span
        className={cn(
          "flex items-center justify-center overflow-hidden rounded-full bg-surface-2 ring-2",
          ring,
        )}
        style={{ width: size, height: size }}
      >
        {imgOk ? (
          <img
            src={MLB.headshot(mlbId)}
            alt=""
            width={size}
            height={size}
            onError={() => setImgOk(false)}
            className="size-full object-cover"
          />
        ) : (
          <span className="font-display text-sm font-bold text-ink-3">{initials}</span>
        )}
      </span>
      {logoOk && (
        <img
          src={MLB.teamLogo(teamId)}
          alt=""
          width={badge}
          height={badge}
          onError={() => setLogoOk(false)}
          className="absolute -bottom-0.5 -right-0.5 rounded-full border border-line bg-white p-0.5"
          style={{ width: badge, height: badge }}
        />
      )}
    </span>
  );
}
