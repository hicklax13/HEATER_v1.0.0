# HEATER Beta — League Onboarding & Launch Playbook

How to invite the 12 FourzynBurn leaguemates to the live HEATER app and link each
login to their team. Written 2026-06-26. The app is beta-ready; this is the last
launch step (per the standing rule: **assigning the leaguemates is done last,
after each has signed in**).

## Live URLs
- **App (what leaguemates use):** https://heater-v1-0-1.vercel.app/
- **API (backend):** https://celebrated-respect-production.up.railway.app (Railway App B `celebrated-respect`)
- Auth: **Clerk** (currently the *dev* instance — fine for the 12-friend beta; the "Development mode" badge is cosmetic. A Clerk *production* instance needs a custom domain and is only required for a public launch.)

## 📩 Invite message (copy-paste to each leaguemate)

> **HEATER is live** — our league's new fantasy app (standings, lineup optimizer, trade analyzer, an AI assistant, all on our real Yahoo data).
>
> 1. Open: **https://heater-v1-0-1.vercel.app/**
> 2. Click **Sign in** → **Continue with Google** (or use your email).
> 3. Done — **text me when you're in.**
>
> Standings, players, and the AI assistant work right away. I'll link your team to your login after you sign in.

## What a leaguemate sees BEFORE they're linked (signed in, no team yet)
- ✅ Works immediately: **Standings, Players, Leaders/Research, Closers, the Bubba AI assistant.**
- 🔒 Personalized pages (Team, Optimizer, Matchup, Trades, Punt) show a friendly **"Your team isn't linked yet"** card until you assign them.
- This is by design (M-3) — the league-wide views don't need a team, so there's value on day one.

## Launch sequence

**Owner (Connor):**
1. Send the invite message to all 11 leaguemates (you = 🏆 Team Hickey, already linked).
2. As each texts "I'm in," note **their name → their team**. No need to wait for all 11 — link them as they trickle in.
3. Tell Claude who has signed in; Claude runs the assignments.

**Claude (when they've signed in):**
1. Read the **Clerk dashboard** Users list (via the owner's signed-in browser) to map each **email → Clerk user id** (`user_…`).
2. With the owner's email→team mapping, `POST /api/admin/assignments` for each (using the owner's admin session — the owner is in `HEATER_ADMIN_CLERK_IDS`).
3. Verify each assignment resolves (re-check `GET /api/admin/assignments`).

## The assignment mechanism (technical reference)

- **Endpoint:** `POST /api/admin/assignments` — body `{ "clerk_user_id": "user_…", "team_name": "<canonical team>" }`. Caller must be an admin (owner's Clerk id is in `HEATER_ADMIN_CLERK_IDS`). Idempotent; pre-provisions the user (works even before their first page load, as long as you have their Clerk id).
- **`team_name`** is reconciled to the canonical Yahoo name (emoji/whitespace/case-tolerant), so passing a recognizable name is fine.
- **Getting each `clerk_user_id`:** the Clerk dashboard (dashboard.clerk.com → the HEATER app → **Users**) lists every signed-in user's email + id. (There is no app endpoint that lists signed-in-but-unassigned users, so the Clerk dashboard is the source of truth for email↔id.)
- **List current assignments / valid team names:** `GET /api/admin/assignments` → `{ assignments, available_teams }`.
- **Reversible:** re-`POST` with a different team to reassign; the membership store overwrites.

## The 12 canonical teams (FourzynBurn)
`BUBBA CROSBY` · `Baty Babies` · `Cyrus The Greats` · `Go yanks` · `Going…Going…Gonorrhea` · `HUMAN INTELLIGENCE` · `Jonny Jockstrap` · `My Precious` · `On a Twosday` · `Over the Rembow` · `The Good The Vlad The Ugly` · `🏆 Team Hickey` (owner — already linked)

**Already assigned:** `🏆 Team Hickey` (conlaxer13@gmail.com).
**Remaining to link (11):** all of the above except Team Hickey.

## Notes
- Bubba (the AI assistant) is live with all 6 providers (Anthropic/OpenAI/Gemini/DeepSeek/xAI/OpenRouter) — managed keys are configured, so leaguemates can use it with no setup.
- Yahoo is **read-only** (HEATER is an advisor; "Apply to Yahoo" lineup writes aren't available — a Yahoo limitation). Users set their final lineup in Yahoo themselves.
- This is the *friends beta* on the current single-league infra. Public/monetized launch (Clerk prod, Postgres, multi-tenancy) is a later phase.
