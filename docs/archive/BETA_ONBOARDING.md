# HEATER Beta Onboarding Guide

Welcome to the HEATER beta! HEATER is an AI-powered fantasy baseball in-season manager built for Head-to-Head Categories leagues on Yahoo Sports. This guide will get you up and running in a few minutes.

---

## Prerequisites

- An active Yahoo Fantasy Baseball league (Head-to-Head Categories format recommended)
- A modern desktop browser (Chrome, Firefox, or Edge — latest two versions)
- Your Yahoo account credentials (for optional live data sync)

---

## Access

- **App URL:** `https://PLACEHOLDER.streamlit.app` *(you will receive the live URL via email)*
- **Access method:** Email-based allowlist — your invite email must match your login
- **Questions:** Email `beta@heaterfantasy.com` if you cannot reach the app

---

## Setup Steps

1. Navigate to the app URL provided in your invite email.
2. On first load, a **Setup Wizard** will appear. Work through each step:
   - Confirm your league format (H2H Categories, 12-team, snake draft).
   - Enter your Yahoo league ID (found in your Yahoo league URL).
   - Optionally connect your Yahoo account via OAuth for live roster and standings sync.
3. After the wizard completes, the app will run a one-time data bootstrap (expect 1-2 minutes on first load — subsequent loads are much faster).
4. You will land on the **My Team** page (Page 1) once bootstrap finishes.

---

## Key Features to Test

Please focus your testing on these four areas:

| Page | Feature | What to Try |
|------|---------|-------------|
| **Page 1 — My Team** | War Room | Check category alerts, mid-week pivot recommendations, hot/cold player flags |
| **Page 3 — Trade Analyzer** | 6-phase trade engine | Build a trade proposal, review the grade breakdown and MC simulation output |
| **Page 5 — Lineup Optimizer** | LP-constrained optimizer | Run the optimizer in Standard mode, review Start/Sit recommendations and streaming picks |
| **Page 11 — Matchup Planner** | Category win probabilities | Check your current matchup, review per-category win probability and daily player detail |

---

## Providing Feedback

**Feedback form (preferred):** [https://forms.gle/PLACEHOLDER](https://forms.gle/PLACEHOLDER)

**Email:** `beta@heaterfantasy.com`

When sharing feedback, please address as many of these as you can:

- Which features did you use most?
- What was confusing or unclear?
- Did any page show an error or produce an obviously wrong recommendation?
- Would you pay for a tool like this, and if so, roughly how much per month?
- What is the single most important feature that is missing?

Screenshots and screen recordings are always welcome and very helpful.

---

## Known Limitations

- **Desktop-optimized:** The app is not designed for mobile or tablet. Use a desktop or laptop browser for the best experience.
- **Yahoo only:** Live data sync requires a Yahoo Fantasy Baseball league. ESPN and CBS leagues can use the app with manual data entry only.
- **Single-user sessions:** Multiple browser tabs open simultaneously on the same account may cause session conflicts. Use one tab at a time.
- **First-load time:** The initial data bootstrap fetches projections, live stats, and league data and typically takes 1-2 minutes. Subsequent visits within the same day are much faster.
- **OAuth token expiry:** If the Yahoo connection drops mid-session, use the "Reconnect Yahoo" button in the sidebar to refresh.

---

*Thank you for helping make HEATER better. Every piece of feedback directly shapes the product.*
