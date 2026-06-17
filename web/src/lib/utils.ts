import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge Tailwind classes safely (clsx + tailwind-merge). */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Legacy humanize-age rule: data is "stale" past 24h (badge turns amber). */
export function humanizeAge(minutes: number): { label: string; stale: boolean } {
  const stale = minutes >= 24 * 60;
  if (minutes < 1) return { label: "just now", stale };
  if (minutes < 60) return { label: `${Math.round(minutes)}m ago`, stale };
  const h = Math.round(minutes / 60);
  if (h < 24) return { label: `${h}h ago`, stale };
  const d = Math.round(h / 24);
  return { label: `${d}d ago`, stale };
}
