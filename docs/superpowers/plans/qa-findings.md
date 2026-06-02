# Per-Team Pre-Launch QA — Findings Log

> Living log of issues found during the per-team QA program. Created Phase 2.
> Severity: **launch_blocker** > **high** > **medium** > **low**.
> Status: **open** → **fixing** → **fixed** (verified) / **wontfix** (design choice).

## How findings are gathered
- **Smoke catalog** (`tests/qa/test_per_team_smoke.py`): every page × every team — crashes / `st.exception` / `st.error`.
- **Deep assertions** (`tests/qa/test_page_*.py`): per-page value plausibility (NaN/None, out-of-range stats, wrong slot counts).
- **silent-failure-hunter**: wrong/empty data that does NOT raise.
- **Browser walkthrough** (Playwright, desktop 1280 + mobile 390): visual/layout/mobile/odd-value.

---

## Phase 2 — Smoke baseline (crash level)

**Run 2026-06-02** (`tests/qa/test_per_team_smoke.py`, serial): **16 passed in 33:23, exit 0.**
- 13 member pages × 12 teams = 156 member renders + 3 admin pages = **159 renders total**.
- **Zero** crashes (`did-not-run`), **zero** `st.exception`, **zero** `st.error` for any (team × page).
- Conclusion: **no launch-blocking crashes** at the page-load level for any of the 12 teams. The deep-assertion suite (value plausibility) and the browser walkthrough are what remain to surface non-crashing / visual issues.

## Findings

| # | Severity | Page | Tab | Team(s) | Problem | Source | Status |
|---|----------|------|-----|---------|---------|--------|--------|
| _none logged yet (smoke clean; deep + browser pending)_ | | | | | | | |

---

## Triage notes
_(adversarial confirmation that each finding is real, not a harness artifact)_

## Resolved / design-choice (not bugs)
_(items confirmed as expected behavior — link to CLAUDE.md "Known Design Choices" where relevant)_
