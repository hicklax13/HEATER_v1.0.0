# HEATER — web (consumer frontend)

Next.js + React + TypeScript + Tailwind v4 port of the Combustion design language.
The legacy Streamlit app is untouched; this lives beside it in `web/`.

## Run

```bash
pnpm -C web dev      # http://localhost:3000
pnpm -C web build    # production build
```

## Stack (verified versions, pinned in package.json)

| Package | Version | Role |
|---|---|---|
| next | 16.2.9 | App Router, RSC |
| react / react-dom | 19.2.x | UI |
| typescript | 5.9.x | types |
| tailwindcss + @tailwindcss/postcss | 4.3.1 | CSS-first theming (`@theme` in `globals.css`) |
| framer-motion | 12.40.0 | count-ups, spring hovers, layout underline, reveal, row drawer |
| @radix-ui/react-dialog · -dropdown-menu · -tooltip · -popover · -tabs | 1.1.x / 2.1.x | accessible primitives |
| cmdk | 1.1.1 | Cmd/K command palette |
| recharts | 3.8.1 | installed for standard charts (signature pieces are hand-rolled SVG) |
| lucide-react | 1.20.0 | line icons |
| clsx · tailwind-merge | 2.1.1 · 3.6.0 | `cn()` |

## Brand tokens

Palette is **cloned from the Gemini brand assets** (`Gemini HEATER TEXT logo_V2.png` +
baseball icon), sampled by `../brand_inbox/_extract_colors*.py`. Defined in
`src/app/globals.css` (`@theme`) + mirrored in `src/lib/tokens.ts`. Navy + tiers seed
from the legacy `THEME`. Fonts self-hosted via `next/font`: Archivo (display, `wdth`
axis on hero numerals), Inter (body), IBM Plex Mono (figures, `tnum` locked).

## Four data-states (per region)

The My Team page exercises all four via `fetchMyTeam()` + a `State` machine in `page.tsx`:

- [x] **loading** — skeleton shimmer mirroring the layout (`LoadingView`)
- [x] **empty** — designed, copy-driven, emoji-free (`EmptyView`)
- [x] **error** — recoverable, with Retry (`ErrorView`)
- [x] **loaded** — full dashboard

## In this first pass

- App shell: logo Home-button, sliding nav underline, **Cmd/K palette**, avatar menu, PRO badge, navy chrome.
- WinHero: **heat-gauge** keyed to value (cool at 46), Archivo `wdth` count-up, drifting hex, labeled win/tie/loss split, machined edge.
- Category outlook: semantic-colored numbers, **polarity-resolved arrows** (ERA handled) + legend, sparklines, win-bars, row drawers, lever↔SB-row link.
- Lever: single primary CTA, pickup tooltips. Movers: fixed slot grammar, headshots, trend, rostered/own%. Ops: IP ring + verdicts. Season trajectory: promoted panel + playoff band + "you are here".
- Motion tokens + `prefers-reduced-motion` (last rule). AA contrast, 44px targets, focus rings, triple-encoded good/bad.

## Deferred to next iteration

- Hero hover-scrub revealing the weekly trajectory inside the gauge.
- recharts adoption for any non-signature charts.
- Per-region (vs page-level) loading/error granularity.
- next/image + `remotePatterns` for MLB headshots (currently plain `<img>`).
