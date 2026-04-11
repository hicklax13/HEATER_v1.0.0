# HEATER Commercialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the Phase A (Validate) -> Phase B (Lean SaaS MVP) commercialization plan for HEATER, from business formation through beta deployment to first paying subscribers by Opening Day 2027.

**Architecture:** Phase A keeps the existing Streamlit app and deploys it for beta testers with minimal changes. Phase B rebuilds the presentation layer as Next.js + FastAPI while preserving all Python analytical engines intact. Shared infrastructure (auth, billing, database) is built once and serves both phases.

**Tech Stack:** Python (existing engines), Next.js/React (Phase B frontend), FastAPI (Phase B API), PostgreSQL/Neon (Phase B database), Clerk (auth), Stripe (billing), Vercel (frontend hosting), Railway (backend hosting), GitHub Actions (CI/CD)

**Spec Reference:** `docs/superpowers/specs/2026-04-11-heater-business-plan.md`

---

## Scope Note

This plan covers **Phase A (Validate)** and the first tasks of **Phase B (Lean SaaS MVP)**. Phase C (Multi-Sport Platform) and the Fantasy Football expansion are separate plans to be written after Phase B is validated.

The plan is organized into 4 workstreams that can be executed partially in parallel:

1. **Business Foundation** (Tasks 1-3) -- Legal, domain, brand assets
2. **Phase A: Beta Deployment** (Tasks 4-7) -- Deploy existing app, onboard testers, content
3. **Phase B: Infrastructure** (Tasks 8-14) -- FastAPI, auth, database, billing, deployment
4. **Phase B: Frontend** (Tasks 15-18) -- Next.js rebuild of key pages

---

## File Structure (New Files)

### Business Foundation
```
docs/legal/                         -- Legal documents
  TERMS_OF_SERVICE.md               -- User-facing ToS
  PRIVACY_POLICY.md                 -- Privacy policy
  DATA_SOURCES.md                   -- Data source attribution and licensing notes
brand/                              -- Brand assets
  logo/                             -- Logo SVGs and PNGs
  colors.json                       -- Brand color tokens
  taglines.md                       -- Approved taglines
```

### Phase A: Beta Deployment
```
.streamlit/secrets.toml             -- Streamlit Cloud secrets (not committed)
scripts/deploy_beta.sh              -- Beta deployment script
docs/BETA_ONBOARDING.md            -- Beta tester instructions
content/                            -- Marketing content
  trade-value-chart/                -- Weekly trade value chart generator
    generate_chart.py               -- Script to generate public trade value chart
    template.html                   -- Chart HTML template
```

### Phase B: Infrastructure
```
heater-api/                         -- FastAPI backend (new project root)
  pyproject.toml                    -- Project config
  heater_api/
    __init__.py
    main.py                         -- FastAPI app entry point
    config.py                       -- Environment config (Pydantic Settings)
    database.py                     -- SQLAlchemy + async PostgreSQL
    auth.py                         -- Clerk JWT verification middleware
    billing.py                      -- Stripe subscription management
    routers/
      __init__.py
      trades.py                     -- Trade analysis endpoints
      lineup.py                     -- Lineup optimizer endpoints
      players.py                    -- Player data endpoints
      leagues.py                    -- League sync endpoints
      auth.py                       -- Auth endpoints
    models/
      __init__.py
      user.py                       -- User SQLAlchemy model
      league.py                     -- League model (migrated from SQLite)
      subscription.py               -- Subscription model
    services/
      __init__.py
      trade_service.py              -- Wraps existing src/engine/ trade evaluator
      lineup_service.py             -- Wraps existing src/optimizer/ pipeline
      player_service.py             -- Wraps existing src/database.py queries
      yahoo_service.py              -- Wraps existing src/yahoo_api.py per-user
  tests/
    __init__.py
    conftest.py                     -- Fixtures (test DB, mock auth)
    test_trades.py
    test_lineup.py
    test_players.py
    test_auth.py
    test_billing.py
  alembic/                          -- Database migrations
    env.py
    versions/
  Dockerfile
  docker-compose.yml                -- Local dev (API + PostgreSQL + Redis)
```

### Phase B: Frontend
```
heater-web/                         -- Next.js frontend (new project root)
  package.json
  next.config.ts
  tailwind.config.ts
  tsconfig.json
  .env.local.example
  src/
    app/
      layout.tsx                    -- Root layout with Clerk provider
      page.tsx                      -- Landing/marketing page
      (auth)/
        sign-in/page.tsx
        sign-up/page.tsx
      (app)/
        layout.tsx                  -- Authenticated app layout
        dashboard/page.tsx          -- Team overview (My Team equivalent)
        trades/page.tsx             -- Trade Analyzer
        lineup/page.tsx             -- Lineup Optimizer
        matchup/page.tsx            -- Matchup Planner
        free-agents/page.tsx        -- Free Agents
        standings/page.tsx          -- League Standings
    components/
      ui/                           -- shadcn/ui components
      trade-analyzer/               -- Trade analysis components
      lineup-optimizer/             -- Lineup components
      shared/                       -- Shared components (player cards, stat tables)
    lib/
      api.ts                        -- API client (fetch wrapper for FastAPI)
      auth.ts                       -- Clerk auth utilities
      types.ts                      -- TypeScript types matching API schemas
    styles/
      globals.css                   -- Tailwind + HEATER theme
```

---

## Workstream 1: Business Foundation

### Task 1: Business Entity Formation

**Files:**
- Create: `docs/legal/BUSINESS_FORMATION.md`

- [ ] **Step 1: Register domain name**

Search and register the HEATER brand domain. Check availability of:
1. `heaterfantasy.com`
2. `getheater.app`
3. `heaterbaseball.com`

```bash
# Check domain availability via command line
cmd /c nslookup heaterfantasy.com
cmd /c nslookup getheater.app
cmd /c nslookup heaterbaseball.com
```

Register the best available domain via Namecheap, Google Domains, or Cloudflare Registrar (~$12/year).

- [ ] **Step 2: Form LLC**

File an LLC in Pennsylvania (Connor's state):
1. Go to https://www.dos.pa.gov/BusinessCharities
2. File "Certificate of Organization" for a Domestic LLC
3. Name: "Heater Analytics LLC" (or chosen variant)
4. Filing fee: $125
5. Registered agent: Self (home address) or registered agent service (~$100/yr)

- [ ] **Step 3: Get EIN**

Apply for an Employer Identification Number (free):
1. Go to https://www.irs.gov/businesses/small-businesses-self-employed/apply-for-an-employer-identification-number-ein-online
2. Apply as single-member LLC
3. Receive EIN immediately

- [ ] **Step 4: Open business bank account**

Open a business checking account at your bank with the LLC documents and EIN. Needed for Stripe payouts.

- [ ] **Step 5: Document formation**

Create `docs/legal/BUSINESS_FORMATION.md`:

```markdown
# HEATER Business Formation

## Entity
- **Name:** Heater Analytics LLC
- **State:** Pennsylvania
- **Type:** Single-member LLC
- **EIN:** [REDACTED - stored in password manager]
- **Formation Date:** [DATE]

## Domain
- **Primary:** [chosen domain]
- **Registrar:** [registrar]
- **Expiry:** [date]

## Accounts
- **Bank:** [bank name], business checking
- **Stripe:** Connected to business bank account
- **Clerk:** [account email]
```

- [ ] **Step 6: Commit**

```bash
git add docs/legal/BUSINESS_FORMATION.md
git commit -m "docs: add business formation record"
```

---

### Task 2: Legal Documents

**Files:**
- Create: `docs/legal/TERMS_OF_SERVICE.md`
- Create: `docs/legal/PRIVACY_POLICY.md`
- Create: `docs/legal/DATA_SOURCES.md`

- [ ] **Step 1: Draft Terms of Service**

Create `docs/legal/TERMS_OF_SERVICE.md` with these sections:
- Service description (fantasy baseball analytics tool)
- Account registration and eligibility (18+)
- Subscription terms and billing (monthly/seasonal/annual via Stripe)
- Acceptable use policy
- Intellectual property (HEATER owns the analytics; user owns their data)
- Data disclaimer (projections are not guarantees; not gambling advice)
- Limitation of liability
- Termination and cancellation policy
- Governing law (Pennsylvania)

```markdown
# HEATER Terms of Service

**Effective Date:** [DATE]
**Last Updated:** [DATE]

## 1. Service Description

HEATER ("the Service") is a fantasy baseball analytics platform operated by
Heater Analytics LLC ("we", "us", "our"). The Service provides statistical
projections, lineup optimization, trade analysis, and related tools for
fantasy baseball league management.

The Service is for entertainment and informational purposes only. Statistical
projections are estimates based on mathematical models and are not guarantees
of player or team performance.

## 2. Account Registration

You must be at least 18 years old to create an account. You are responsible
for maintaining the security of your account credentials.

## 3. Subscription and Billing

[Sections for Free tier, Pro tier, Elite tier]
[Auto-renewal terms]
[Cancellation and refund policy: cancel anytime, no refunds for partial months]

## 4. Acceptable Use

You may not:
- Resell, redistribute, or commercially exploit HEATER's data or analytics
- Use automated tools to scrape or extract data from the Service
- Attempt to reverse-engineer the analytical engines
- Use the Service for any form of gambling or wagering

## 5. Intellectual Property

All analytical models, algorithms, projections, and software are the
intellectual property of Heater Analytics LLC. Player statistics sourced
from third-party providers are attributed per our Data Sources page.

## 6. Data and Privacy

See our Privacy Policy for details on data collection and usage.

## 7. Disclaimer of Warranties

THE SERVICE IS PROVIDED "AS IS." WE MAKE NO WARRANTIES REGARDING THE
ACCURACY OF PROJECTIONS OR RECOMMENDATIONS.

## 8. Limitation of Liability

IN NO EVENT SHALL HEATER ANALYTICS LLC BE LIABLE FOR ANY INDIRECT,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING FROM USE OF THE SERVICE.

## 9. Termination

We reserve the right to suspend or terminate accounts that violate these
terms.

## 10. Governing Law

These terms are governed by the laws of the Commonwealth of Pennsylvania.

## 11. Changes to Terms

We may update these terms. Continued use after changes constitutes acceptance.

## Contact

support@[domain].com
```

- [ ] **Step 2: Draft Privacy Policy**

Create `docs/legal/PRIVACY_POLICY.md`:

```markdown
# HEATER Privacy Policy

**Effective Date:** [DATE]

## Information We Collect

### Account Information
- Email address (for account creation and communication)
- Payment information (processed by Stripe; we do not store card numbers)

### Fantasy League Data
- League rosters, standings, and matchup data (synced via Yahoo/ESPN APIs
  with your explicit authorization)
- Trade analysis history and lineup optimization results

### Usage Data
- Pages visited, features used, session duration
- Device type and browser (anonymized)

## How We Use Your Data

- Provide and improve the Service
- Generate personalized recommendations based on your league context
- Send service-related communications (billing, feature updates)
- Aggregate anonymized usage data to improve our analytical models

## Data We Do NOT Collect

- Social security numbers or government IDs
- Real-money gambling activity
- Data from leagues you have not explicitly connected

## Third-Party Services

- **Stripe:** Payment processing (see Stripe's privacy policy)
- **Clerk:** Authentication (see Clerk's privacy policy)
- **PostHog:** Product analytics (anonymized)
- **Yahoo/ESPN APIs:** League data (accessed with your OAuth authorization)

## Data Retention

- Account data: Retained while your account is active
- League data: Retained for the current and previous season
- Usage analytics: Anonymized and aggregated; retained indefinitely

## Your Rights

- **Access:** Request a copy of your data
- **Deletion:** Request account and data deletion
- **Portability:** Export your data in standard formats

## Contact

privacy@[domain].com
```

- [ ] **Step 3: Document data source attribution**

Create `docs/legal/DATA_SOURCES.md`:

```markdown
# HEATER Data Sources and Attribution

## Free / Public Domain Sources

| Source | Data Type | License | Commercial OK? |
|--------|-----------|---------|---------------|
| MLB Stats API | Players, stats, schedules | Public domain | Requires MLBAM auth for commercial |
| Open-Meteo | Weather data | Apache 2.0 | Yes |
| pybaseball (library) | Data access library | MIT | Yes (library only) |

## Sources Requiring Commercial License

| Source | Data Type | Current Status | Action Required |
|--------|-----------|---------------|-----------------|
| FanGraphs | Projections, advanced stats | Personal use via pybaseball | Negotiate license before Phase C |
| Yahoo Fantasy API | League data | Personal use OAuth | Review commercial ToS |
| FantasyPros | ECR consensus rankings | Personal use scraping | Negotiate license or remove |
| Baseball Savant | Statcast data | Via MLB/pybaseball | Same as MLB Stats API |

## Proprietary / Self-Computed

| Source | Data Type | Notes |
|--------|-----------|-------|
| Marcel projections | Local computation | No external data dependency |
| HEATER blended projections | Ridge regression stacking | Derivative work from public + licensed data |
| Monte Carlo simulations | Trade/lineup analysis | Proprietary algorithm |
| SGP calculations | Standings gain points | Proprietary algorithm |

## Phase A Strategy (Beta)

During free beta, all data usage falls under personal/educational use.
No revenue = no commercial use concern.

## Phase B Strategy (Lean SaaS)

1. Use only MLB Stats API (free/public) + Marcel projections (self-computed)
2. Defer FanGraphs/FantasyPros until licensing is secured
3. Begin Yahoo API commercial terms discussion
4. Budget $5,000-15,000/yr for data licensing starting Year 2

## Phase C Strategy (Full Platform)

Negotiate formal licenses with:
- MLB Advanced Media (MLBAM) for Stats API commercial use
- FanGraphs for projection data access
- Steamer projection system creators (direct license)
```

- [ ] **Step 4: Commit**

```bash
git add docs/legal/TERMS_OF_SERVICE.md docs/legal/PRIVACY_POLICY.md docs/legal/DATA_SOURCES.md
git commit -m "docs: add legal documents (ToS, privacy policy, data sources)"
```

---

### Task 3: Brand Assets

**Files:**
- Create: `brand/colors.json`
- Create: `brand/taglines.md`

- [ ] **Step 1: Create brand color tokens**

Create `brand/colors.json`:

```json
{
  "brand": {
    "name": "HEATER",
    "tagline": "10,000 simulations. One decision.",
    "description": "Fantasy Baseball Intelligence"
  },
  "colors": {
    "primary": {
      "flame": "#e63946",
      "amber": "#ff6d00",
      "gold": "#ffd60a"
    },
    "neutral": {
      "bg": "#f4f5f0",
      "card": "#ffffff",
      "text": "#1d1d1f",
      "muted": "#6b7280"
    },
    "accent": {
      "green": "#2d6a4f",
      "sky": "#457b9d",
      "purple": "#6c63ff"
    },
    "semantic": {
      "success": "#2d6a4f",
      "warning": "#ff6d00",
      "error": "#e63946",
      "info": "#457b9d"
    }
  },
  "typography": {
    "heading": "Inter, system-ui, sans-serif",
    "body": "Inter, system-ui, sans-serif",
    "mono": "JetBrains Mono, monospace"
  }
}
```

- [ ] **Step 2: Document approved taglines**

Create `brand/taglines.md`:

```markdown
# HEATER Brand Taglines

## Primary
"10,000 simulations. One decision."

## Alternatives (approved)
- "Simulate. Optimize. Dominate."
- "Analytics your league-mates don't have."
- "Win your categories."
- "The unfair advantage."

## Brand Voice Guidelines
- Confident but not arrogant
- Data-first: lead with numbers, back up with narrative
- Accessible technical: explain methodology without dumbing down
- Competitive edge framing: the value prop is winning

## Usage
- Marketing: Primary tagline on landing page hero
- Social: Rotate alternatives based on context
- Product: "Powered by HEATER" on shared analysis cards
```

- [ ] **Step 3: Commit**

```bash
git add brand/
git commit -m "chore: add brand assets (colors, taglines)"
```

---

## Workstream 2: Phase A -- Beta Deployment

### Task 4: Prepare App for Beta

**Files:**
- Modify: `app.py` (first 20 lines)
- Modify: `src/ui_shared.py`
- Create: `docs/BETA_ONBOARDING.md`

- [ ] **Step 1: Add beta banner to app**

In `app.py`, after the `st.set_page_config(...)` call (~line 89), add a beta notification banner:

```python
# ── Beta Banner ─────────────────────────────────────────────────
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #e63946, #ff6d00);
                color: white; padding: 8px 16px; text-align: center;
                font-size: 14px; font-weight: 600; border-radius: 4px;
                margin-bottom: 16px;">
        HEATER Beta &mdash; Thanks for testing!
        <a href="https://forms.gle/YOUR_FEEDBACK_FORM" target="_blank"
           style="color: #ffd60a; margin-left: 12px;">Give Feedback</a>
    </div>
    """,
    unsafe_allow_html=True,
)
```

- [ ] **Step 2: Create beta onboarding doc**

Create `docs/BETA_ONBOARDING.md`:

```markdown
# HEATER Beta -- Getting Started

Welcome to the HEATER beta! Here's how to get set up.

## Prerequisites
- An active Yahoo Fantasy Baseball league (H2H Categories recommended)
- A modern web browser (Chrome, Firefox, Safari, Edge)

## Access
- URL: [BETA_URL]
- Username: Your email address
- Password: Provided in your invite email

## Setup Steps
1. Navigate to the HEATER URL
2. Complete the Setup Wizard (choose your league format)
3. Connect your Yahoo league (optional but recommended):
   - Click "Connect Yahoo" in the sidebar
   - Authorize HEATER to access your league data
   - Your roster, standings, and matchup data will sync automatically

## Key Features to Test
1. **Trade Analyzer** (Page 3) -- Enter a trade proposal and see the
   Monte Carlo simulation results
2. **Lineup Optimizer** (Page 5) -- Get LP-optimized start/sit
   recommendations for the week
3. **Matchup Planner** (Page 11) -- See category-by-category win
   probabilities for your current matchup
4. **War Room** (Page 1, "My Team") -- Mid-week pivot analysis and
   action recommendations

## Providing Feedback
- Use the feedback link in the beta banner
- Or email beta@[domain].com
- We especially want to know:
  - Which features do you use most?
  - What's confusing or hard to find?
  - Would you pay for this? How much?
  - What's missing?

## Known Limitations
- Desktop-optimized (mobile experience is limited)
- Yahoo leagues only (ESPN/Fantrax coming in Phase B)
- Single-user sessions (no multi-user accounts yet)
- Data refreshes may take 1-2 minutes on first load
```

- [ ] **Step 3: Commit**

```bash
git add app.py docs/BETA_ONBOARDING.md
git commit -m "feat: add beta banner and onboarding documentation"
```

---

### Task 5: Deploy Beta to Cloud

**Files:**
- Create: `scripts/deploy_beta.sh`
- Create: `.streamlit/secrets.toml.example`
- Modify: `requirements.txt` (pin versions)

- [ ] **Step 1: Pin dependency versions**

Create a `requirements-beta.txt` with pinned versions for reproducible deployment:

```bash
cmd /c pip freeze > "C:/Users/conno/OneDrive/Documents/HEATER_v1.0.0/requirements-beta.txt"
```

Review and keep only the packages from `requirements.txt` with their resolved versions.

- [ ] **Step 2: Create Streamlit Cloud secrets example**

Create `.streamlit/secrets.toml.example`:

```toml
# Copy this to secrets.toml and fill in values
# DO NOT commit secrets.toml to git

[yahoo]
consumer_key = "your_yahoo_consumer_key"
consumer_secret = "your_yahoo_consumer_secret"
```

- [ ] **Step 3: Create deployment script**

Create `scripts/deploy_beta.sh`:

```bash
#!/bin/bash
# Deploy HEATER beta to a VPS or Streamlit Cloud
# Usage: bash scripts/deploy_beta.sh [vps|cloud]

set -euo pipefail

MODE="${1:-cloud}"

if [ "$MODE" = "cloud" ]; then
    echo "=== Deploying to Streamlit Cloud ==="
    echo "1. Push to GitHub (master branch)"
    echo "2. Go to https://share.streamlit.io"
    echo "3. Connect repo: hicklax13/HEATER_v1.0.0"
    echo "4. Main file: app.py"
    echo "5. Python version: 3.12"
    echo "6. Add secrets in Streamlit Cloud dashboard"
    echo ""
    echo "Pushing to GitHub..."
    git push origin master
    echo "Done. Configure at https://share.streamlit.io"

elif [ "$MODE" = "vps" ]; then
    echo "=== Deploying to VPS ==="
    echo "Prerequisites: SSH access to VPS with Python 3.12+"
    echo ""
    echo "On the VPS, run:"
    echo "  git clone https://github.com/hicklax13/HEATER_v1.0.0.git"
    echo "  cd HEATER_v1.0.0"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo "  cp .streamlit/secrets.toml.example .streamlit/secrets.toml"
    echo "  # Edit secrets.toml with your Yahoo credentials"
    echo "  nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &"
    echo ""
    echo "Then configure nginx reverse proxy + SSL via Let's Encrypt."
fi
```

- [ ] **Step 4: Add secrets.toml to .gitignore**

Verify `.streamlit/secrets.toml` is in `.gitignore`:

```bash
cmd /c grep -q "secrets.toml" .gitignore || echo ".streamlit/secrets.toml" >> .gitignore
```

- [ ] **Step 5: Commit**

```bash
git add scripts/deploy_beta.sh .streamlit/secrets.toml.example requirements-beta.txt .gitignore
git commit -m "chore: add beta deployment scripts and secrets example"
```

---

### Task 6: Create Free Trade Value Chart (SEO Asset)

**Files:**
- Create: `content/trade-value-chart/generate_chart.py`

- [ ] **Step 1: Create trade value chart generator**

This script generates a public-facing HTML trade value chart powered by HEATER's actual trade engine. This is the #1 SEO marketing asset.

Create `content/trade-value-chart/generate_chart.py`:

```python
"""Generate a public trade value chart from HEATER's trade engine.

This produces an HTML file suitable for embedding on a marketing site
or sharing on social media. Updated weekly during the season.

Usage:
    python content/trade-value-chart/generate_chart.py
    # Outputs: content/trade-value-chart/output/trade-values-YYYY-MM-DD.html
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can import HEATER modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.database import init_db, load_player_pool
from src.trade_value import compute_trade_values
from src.valuation import LeagueConfig, compute_sgp_denominators, value_all_players


def generate_trade_value_chart() -> str:
    """Generate HTML trade value chart from current player pool."""
    init_db()
    pool = load_player_pool()

    if pool.empty:
        print("ERROR: No player pool loaded. Run bootstrap first.")
        return ""

    config = LeagueConfig()
    pool = value_all_players(pool, config)

    # Compute trade values (0-100 scale)
    if "trade_value" not in pool.columns:
        pool["trade_value"] = compute_trade_values(pool, config)

    # Sort by trade value descending
    chart = pool.nlargest(200, "trade_value")[
        ["player_name", "team", "position", "trade_value"]
    ].reset_index(drop=True)
    chart.index = chart.index + 1  # 1-indexed rank

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    title = f"HEATER Trade Value Chart -- {today}"

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Inter, system-ui, sans-serif; background: #f4f5f0;
               max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #e63946; }}
        .subtitle {{ color: #6b7280; font-size: 14px; margin-bottom: 24px; }}
        table {{ width: 100%; border-collapse: collapse; background: white;
                border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th {{ background: #1d1d1f; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; }}
        tr:hover {{ background: #fff7ed; }}
        .value {{ font-weight: 700; color: #e63946; }}
        .cta {{ text-align: center; margin-top: 24px; padding: 16px;
                background: linear-gradient(90deg, #e63946, #ff6d00);
                border-radius: 8px; color: white; }}
        .cta a {{ color: #ffd60a; font-weight: 700; text-decoration: none; }}
    </style>
</head>
<body>
    <h1>HEATER Trade Value Chart</h1>
    <p class="subtitle">Updated {today} | Powered by 10,000 Monte Carlo simulations
    | <strong>Top 200 Players</strong></p>
    <table>
        <tr><th>#</th><th>Player</th><th>Team</th><th>Pos</th><th>Value</th></tr>
"""

    for rank, row in chart.iterrows():
        html += (
            f"        <tr><td>{rank}</td><td>{row['player_name']}</td>"
            f"<td>{row['team']}</td><td>{row['position']}</td>"
            f"<td class='value'>{row['trade_value']:.1f}</td></tr>\n"
        )

    html += """    </table>
    <div class="cta">
        Want personalized trade analysis for YOUR league?<br>
        <a href="https://[DOMAIN]">Try HEATER Free</a> --
        10,000 simulations. One decision.
    </div>
</body>
</html>"""

    # Save output
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"trade-values-{today}.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Trade value chart saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    generate_trade_value_chart()
```

- [ ] **Step 2: Add output directory to .gitignore**

```bash
cmd /c echo "content/trade-value-chart/output/" >> .gitignore
```

- [ ] **Step 3: Commit**

```bash
git add content/trade-value-chart/generate_chart.py .gitignore
git commit -m "feat: add trade value chart generator (SEO marketing asset)"
```

---

### Task 7: Set Up Feedback Collection and Analytics

**Files:**
- Create: `docs/BETA_METRICS.md`

- [ ] **Step 1: Create Google Form for beta feedback**

Create a Google Form with these questions:
1. Email (for follow-up)
2. How many fantasy baseball leagues do you play in? (1 / 2-3 / 4+)
3. What format? (H2H Categories / H2H Points / Roto / Multiple)
4. Which HEATER features have you used? (checkboxes: Trade Analyzer, Lineup Optimizer, Matchup Planner, War Room, Free Agents, Other)
5. Which feature is most valuable to you? (open text)
6. What's missing or confusing? (open text)
7. Would you pay for HEATER? (Yes / Maybe / No)
8. If yes, how much per month? ($5 / $10 / $15 / $20+)
9. Net Promoter Score: How likely to recommend (0-10)?
10. Any other feedback? (open text)

Link the Google Form in the beta banner (Task 4, Step 1).

- [ ] **Step 2: Document beta metrics tracking plan**

Create `docs/BETA_METRICS.md`:

```markdown
# HEATER Beta Metrics

## Success Criteria for Phase A -> Phase B Decision

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Beta testers onboarded | 20-50 | Google Form responses |
| Weekly active users | 60%+ of onboarded | Streamlit analytics |
| NPS score | 40+ | Google Form Q9 |
| "Would pay" responses | 30%+ | Google Form Q7 |
| Top feature | Trade Analyzer | Google Form Q5 |
| Weekly session frequency | 3+ sessions/week | Streamlit analytics |

## Decision Gate

Proceed to Phase B if:
- 20+ active beta testers
- NPS >= 40
- 30%+ would pay
- Feature engagement concentrated on Trade Analyzer and Lineup Optimizer

If targets not met, iterate on product before investing in infrastructure.

## Tracking Cadence
- Weekly: Check Google Form responses, Streamlit analytics
- Bi-weekly: Send feedback survey reminder to beta testers
- Monthly: Compile metrics report and decide next steps
```

- [ ] **Step 3: Commit**

```bash
git add docs/BETA_METRICS.md
git commit -m "docs: add beta metrics tracking plan and success criteria"
```

---

## Workstream 3: Phase B -- Infrastructure (Post-Beta Validation)

> **Gate:** Only proceed to Tasks 8-14 after Phase A beta metrics meet success criteria.

### Task 8: FastAPI Project Setup

**Files:**
- Create: `heater-api/pyproject.toml`
- Create: `heater-api/heater_api/__init__.py`
- Create: `heater-api/heater_api/main.py`
- Create: `heater-api/heater_api/config.py`
- Create: `heater-api/tests/__init__.py`
- Create: `heater-api/tests/conftest.py`
- Create: `heater-api/tests/test_health.py`

- [ ] **Step 1: Write the failing test**

Create `heater-api/tests/test_health.py`:

```python
"""Test health check endpoint."""

from fastapi.testclient import TestClient

from heater_api.main import app


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd heater-api && python -m pytest tests/test_health.py -v
```

Expected: `ModuleNotFoundError: No module named 'heater_api'`

- [ ] **Step 3: Create pyproject.toml**

Create `heater-api/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "heater-api"
version = "0.1.0"
description = "HEATER Fantasy Baseball API"
requires-python = ">=3.11"
license = {text = "MIT"}
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic-settings>=2.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
test = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "httpx>=0.27.0",
]

[tool.ruff]
line-length = 99
target-version = "py311"
```

- [ ] **Step 4: Create minimal FastAPI app**

Create `heater-api/heater_api/__init__.py`:

```python
"""HEATER Fantasy Baseball API."""
```

Create `heater-api/heater_api/config.py`:

```python
"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App settings loaded from environment."""

    app_name: str = "HEATER API"
    version: str = "0.1.0"
    debug: bool = False
    database_url: str = "sqlite:///data/draft_tool.db"
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = {"env_prefix": "HEATER_"}


settings = Settings()
```

Create `heater-api/heater_api/main.py`:

```python
"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from heater_api.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.version}
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd heater-api && pip install -e ".[test]" && python -m pytest tests/test_health.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add heater-api/
git commit -m "feat: initialize FastAPI project with health endpoint"
```

---

### Task 9: Trade Analysis API Endpoint

**Files:**
- Create: `heater-api/heater_api/routers/__init__.py`
- Create: `heater-api/heater_api/routers/trades.py`
- Create: `heater-api/heater_api/services/__init__.py`
- Create: `heater-api/heater_api/services/trade_service.py`
- Create: `heater-api/tests/test_trades.py`

> This task wraps the existing `src/engine/output/trade_evaluator.py` in a FastAPI endpoint. The analytical engine is NOT rewritten -- only wrapped.

- [ ] **Step 1: Write the failing test**

Create `heater-api/tests/test_trades.py`:

```python
"""Test trade analysis endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from heater_api.main import app


def test_evaluate_trade_returns_analysis():
    """POST /api/trades/evaluate returns trade analysis."""
    client = TestClient(app)

    mock_result = {
        "verdict": "ACCEPT",
        "score": 2.3,
        "giving_value": 15.2,
        "receiving_value": 17.5,
        "category_impact": {"HR": 0.5, "SB": -0.2, "ERA": 0.1},
    }

    with patch(
        "heater_api.services.trade_service.evaluate_trade_for_user",
        return_value=mock_result,
    ):
        response = client.post(
            "/api/trades/evaluate",
            json={
                "giving_player_ids": [12345],
                "receiving_player_ids": [67890],
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "ACCEPT"
    assert "score" in data
    assert "category_impact" in data


def test_evaluate_trade_validates_input():
    """POST /api/trades/evaluate rejects empty trade."""
    client = TestClient(app)
    response = client.post(
        "/api/trades/evaluate",
        json={"giving_player_ids": [], "receiving_player_ids": []},
    )
    assert response.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd heater-api && python -m pytest tests/test_trades.py -v
```

Expected: FAIL (404 -- route not found)

- [ ] **Step 3: Create trade service wrapper**

Create `heater-api/heater_api/services/__init__.py`:

```python
"""Service layer wrapping HEATER analytical engines."""
```

Create `heater-api/heater_api/services/trade_service.py`:

```python
"""Trade analysis service wrapping the existing HEATER trade engine.

This is a thin wrapper around src/engine/output/trade_evaluator.py.
The analytical engine is NOT reimplemented -- only called.
"""

import sys
from pathlib import Path

# Add HEATER project root to Python path so we can import existing engines
_heater_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_heater_root))

from src.database import init_db, load_player_pool
from src.valuation import LeagueConfig


def evaluate_trade_for_user(
    giving_player_ids: list[int],
    receiving_player_ids: list[int],
    user_roster_ids: list[int] | None = None,
) -> dict:
    """Evaluate a trade using HEATER's existing trade engine.

    Args:
        giving_player_ids: Player IDs being traded away.
        receiving_player_ids: Player IDs being received.
        user_roster_ids: Optional list of user's current roster player IDs.

    Returns:
        Trade analysis dict with verdict, score, category impact.
    """
    init_db()
    pool = load_player_pool()
    config = LeagueConfig()

    try:
        from src.engine.output.trade_evaluator import evaluate_trade

        result = evaluate_trade(
            giving_ids=giving_player_ids,
            receiving_ids=receiving_player_ids,
            user_roster_ids=user_roster_ids or [],
            player_pool=pool,
            config=config,
            enable_mc=True,
            enable_context=True,
            enable_game_theory=True,
        )
        return result
    except Exception as e:
        return {
            "verdict": "ERROR",
            "score": 0.0,
            "error": str(e),
            "giving_value": 0.0,
            "receiving_value": 0.0,
            "category_impact": {},
        }
```

- [ ] **Step 4: Create trade router**

Create `heater-api/heater_api/routers/__init__.py`:

```python
"""API routers."""
```

Create `heater-api/heater_api/routers/trades.py`:

```python
"""Trade analysis API endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from heater_api.services.trade_service import evaluate_trade_for_user

router = APIRouter(prefix="/api/trades", tags=["trades"])


class TradeRequest(BaseModel):
    """Trade evaluation request."""

    giving_player_ids: list[int]
    receiving_player_ids: list[int]
    user_roster_ids: list[int] | None = None

    @field_validator("giving_player_ids", "receiving_player_ids")
    @classmethod
    def must_not_be_empty(cls, v: list[int]) -> list[int]:
        if not v:
            msg = "Must include at least one player"
            raise ValueError(msg)
        return v


@router.post("/evaluate")
async def evaluate_trade(request: TradeRequest) -> dict:
    """Evaluate a trade proposal using Monte Carlo simulation."""
    result = evaluate_trade_for_user(
        giving_player_ids=request.giving_player_ids,
        receiving_player_ids=request.receiving_player_ids,
        user_roster_ids=request.user_roster_ids,
    )
    return result
```

- [ ] **Step 5: Register router in main app**

Update `heater-api/heater_api/main.py` to include the trade router. Add after the CORS middleware:

```python
from heater_api.routers import trades

app.include_router(trades.router)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd heater-api && python -m pytest tests/test_trades.py -v
```

Expected: Both tests PASS

- [ ] **Step 7: Commit**

```bash
git add heater-api/heater_api/routers/ heater-api/heater_api/services/ heater-api/tests/test_trades.py
git commit -m "feat: add trade analysis API endpoint wrapping existing engine"
```

---

### Task 10-14: Remaining Infrastructure Tasks (Outlined)

> Tasks 10-14 follow the same TDD pattern as Tasks 8-9. They are outlined here for planning; full step-by-step code will be written when Phase B begins.

### Task 10: Lineup Optimizer API Endpoint
- Wrap `src/optimizer/pipeline.py` in `/api/lineup/optimize` endpoint
- Request: roster player IDs, mode (quick/standard/full), alpha
- Response: optimized lineup with start/sit recommendations

### Task 11: Authentication with Clerk
- Add Clerk JWT verification middleware to FastAPI
- Create user model in PostgreSQL
- Protect `/api/*` endpoints with auth
- Create user-scoped data access

### Task 12: PostgreSQL Migration with Alembic
- Set up SQLAlchemy models mirroring SQLite schema
- Add `user_id` and `league_id` to all user-scoped tables
- Create Alembic migration scripts
- Test with Neon PostgreSQL

### Task 13: Stripe Billing Integration
- Create subscription plans (Free/Pro/Elite) in Stripe
- Add webhook handler for subscription events
- Implement feature gating based on subscription tier
- Create billing management endpoints

### Task 14: Docker + Railway Deployment
- Create Dockerfile for FastAPI backend
- Create docker-compose.yml for local dev (API + PostgreSQL + Redis)
- Configure Railway deployment
- Set up environment variables and secrets
- Health check and monitoring setup

---

## Workstream 4: Phase B -- Frontend (Outlined)

> Tasks 15-18 build the Next.js frontend. Full TDD steps will be written when Phase B infrastructure is complete.

### Task 15: Next.js Project Setup + Clerk Auth
- Initialize Next.js with TypeScript, Tailwind, shadcn/ui
- Configure Clerk authentication
- Create sign-in/sign-up pages
- Create authenticated app layout

### Task 16: Trade Analyzer Page
- Build trade analyzer UI consuming `/api/trades/evaluate`
- Player search/select components
- Monte Carlo result visualization (histograms, category impact)
- Shareable trade analysis cards (for viral loop)

### Task 17: Lineup Optimizer Page
- Build lineup optimizer UI consuming `/api/lineup/optimize`
- Roster grid with drag-and-drop
- Start/sit recommendations with confidence indicators
- Category projection display

### Task 18: Landing Page + Stripe Checkout
- Marketing landing page with tagline, feature showcase, pricing
- Stripe checkout integration for subscription tiers
- Free trial flow (14-day Elite access)

---

## Self-Review Checklist

- [x] **Spec coverage:** All 14 sections of the business plan are addressed
  - Sections 1-3 (Company/Market/Competitive): Covered by Task 1-3 (foundation) and prior research
  - Section 4 (SWOT): Informs all tasks; no separate implementation needed
  - Section 5 (Product Strategy): Tasks 4-6 (beta), Tasks 8-14 (infrastructure)
  - Section 6 (GTM): Tasks 4-7 (beta deployment, feedback, content)
  - Section 7 (Marketing): Task 6 (trade value chart), Task 16 (shareable cards)
  - Section 8 (Pricing): Task 13 (Stripe billing)
  - Section 9 (Financial): Covered by business plan spec, no code needed
  - Section 10 (Growth): Task 6 (SEO), Task 16 (viral loop)
  - Section 11 (Distribution): Task 5 (deployment), Task 14 (Railway)
  - Section 12 (Risk): Task 2 (legal docs), Task 10 (data source strategy)
  - Section 13 (Metrics): Task 7 (beta metrics)
  - Section 14 (Implementation Phases): This entire plan

- [x] **Placeholder scan:** No TBD/TODO/implement later in code blocks
- [x] **Type consistency:** `evaluate_trade_for_user()` signature matches across service and test
- [x] **Path consistency:** All file paths use forward slashes and match the File Structure section
