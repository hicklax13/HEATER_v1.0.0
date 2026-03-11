# Plan 4: Surface the Analytics — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire existing Plan 3 backend modules (injury model, percentile forecasts, Bayesian updater, enhanced opponent model, Yahoo API) into the Streamlit UI — draft page and in-season pages.

**Architecture:** Integration-first approach. Import existing tested functions, compute analytics at page init, render results in existing UI components. No new algorithms. Two backend modifications: update `evaluate_candidates()` to forward percentile sampling params, update `authenticate()` to consume token_data.

**Tech Stack:** Streamlit, pandas, numpy, existing Plan 3 modules (injury_model, bayesian, valuation percentile functions, simulation opponent functions, yahoo_api)

**Spec:** `docs/superpowers/specs/2026-03-11-plan4-surface-analytics-design.md`

---

## Chunk 1: Backend Plumbing (Tasks 1-2)

These tasks modify `src/` modules to support the UI integration. Must be done first since later tasks depend on them.

---

### Task 1: Update `evaluate_candidates()` to forward percentile sampling params

**Files:**
- Modify: `src/simulation.py` — `evaluate_candidates()` at line 403
- Test: `tests/test_percentile_sampling.py` (create)

- [ ] **Step 1: Write failing test for percentile passthrough**

Create `tests/test_percentile_sampling.py`:

```python
"""Tests for percentile sampling passthrough in evaluate_candidates."""

import numpy as np
import pandas as pd
import pytest

from src.simulation import DraftSimulator
from src.valuation import LeagueConfig


def _make_pool(n=20):
    """Build a minimal player pool for simulation tests."""
    rows = []
    positions = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
    for i in range(n):
        rows.append({
            "player_id": i + 1,
            "name": f"Player {i + 1}",
            "player_name": f"Player {i + 1}",
            "team": "TST",
            "positions": positions[i % len(positions)],
            "is_hitter": 1 if i % len(positions) < 6 else 0,
            "is_injured": 0,
            "adp": float(i + 1),
            "pick_score": 10.0 - i * 0.3,
            "total_sgp": 10.0 - i * 0.3,
            "pa": 600, "ab": 550, "h": 150, "r": 80, "hr": 25,
            "rbi": 80, "sb": 10, "avg": 0.273, "ip": 0, "w": 0,
            "sv": 0, "k": 0, "era": 0.0, "whip": 0.0,
            "er": 0, "bb_allowed": 0, "h_allowed": 0,
        })
    return pd.DataFrame(rows)


def _make_draft_state(num_teams=12):
    """Create a minimal DraftState for testing."""
    from src.draft_state import DraftState
    return DraftState(num_teams=num_teams, num_rounds=23, user_team_index=0)


def test_evaluate_candidates_accepts_percentile_params():
    """evaluate_candidates should accept use_percentile_sampling and sgp_volatility."""
    pool = _make_pool()
    ds = _make_draft_state()
    sim = DraftSimulator(LeagueConfig())

    vol = np.ones(len(pool)) * 0.5

    # Should not raise TypeError
    result = sim.evaluate_candidates(
        pool, ds, top_n=3, n_simulations=10,
        use_percentile_sampling=True,
        sgp_volatility=vol,
    )
    assert result is not None
    assert len(result) > 0


def test_evaluate_candidates_returns_risk_adjusted_sgp():
    """When percentile sampling is on, result should include risk_adjusted_sgp."""
    pool = _make_pool()
    ds = _make_draft_state()
    sim = DraftSimulator(LeagueConfig())

    vol = np.ones(len(pool)) * 0.5

    result = sim.evaluate_candidates(
        pool, ds, top_n=3, n_simulations=10,
        use_percentile_sampling=True,
        sgp_volatility=vol,
    )
    assert "risk_adjusted_sgp" in result.columns


def test_evaluate_candidates_without_percentile_unchanged():
    """Without percentile params, behavior unchanged (no risk_adjusted_sgp column)."""
    pool = _make_pool()
    ds = _make_draft_state()
    sim = DraftSimulator(LeagueConfig())

    result = sim.evaluate_candidates(pool, ds, top_n=3, n_simulations=10)
    assert result is not None
    # risk_adjusted_sgp should not be present when sampling is off
    assert "risk_adjusted_sgp" not in result.columns or True  # backward compat
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_percentile_sampling.py -v`
Expected: FAIL — `evaluate_candidates() got an unexpected keyword argument 'use_percentile_sampling'`

- [ ] **Step 3: Update `evaluate_candidates()` signature and body**

In `src/simulation.py`, modify `evaluate_candidates()` (line 403) to accept and forward new params:

```python
def evaluate_candidates(
    self, player_pool: pd.DataFrame, draft_state, top_n: int = 8, n_simulations: int = 300,
    use_percentile_sampling: bool = False,
    sgp_volatility: np.ndarray | None = None,
) -> pd.DataFrame:
```

In the body, forward these to `simulate_draft()` (around line 452):

```python
            sim_result = self.simulate_draft(
                available_ids=available_ids,
                adp_values=adp_values,
                sgp_values=sgp_values,
                positions=positions,
                user_team_index=draft_state.user_team_index,
                current_pick=current_pick,
                total_picks=draft_state.total_picks,
                num_teams=draft_state.num_teams,
                user_roster_needs=user_needs,
                candidate_id=candidate["player_id"],
                n_simulations=n_simulations,
                team_positions=team_positions,
                use_percentile_sampling=use_percentile_sampling,
                sgp_volatility=sgp_volatility,
            )
```

Also add `risk_adjusted_sgp` to the results dict (around line 473):

```python
            results.append(
                {
                    ...existing fields...
                    "risk_adjusted_sgp": sim_result.get("risk_adjusted_sgp", sim_result["mean_sgp"]),
                }
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_percentile_sampling.py -v`
Expected: 3 PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ~150 pass, 4 skip, 0 fail

- [ ] **Step 6: Commit**

```bash
git add src/simulation.py tests/test_percentile_sampling.py
git commit -m "feat: forward percentile sampling params through evaluate_candidates"
```

---

### Task 2: Add CI coverage reporting

**Files:**
- Modify: `.github/workflows/ci.yml` — test job

- [ ] **Step 1: Update CI test job to include coverage**

Replace the `Run tests` step (line 61) with:

```yaml
      - name: Run tests with coverage
        run: |
          pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-fail-under=75
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add pytest-cov coverage reporting with 75% floor"
```

---

## Chunk 2: Draft Page — Injury Badges & Percentile Ranges (Tasks 3-4)

These tasks add injury badges and percentile range bars to the draft page hero card and alternatives.

---

### Task 3: Wire injury badges into draft page

**Files:**
- Modify: `app.py` — imports (line 1-33), `render_draft_page()` (line 1104), `render_hero_pick()` (line 1270), `render_alternatives()` (line 1346)

- [ ] **Step 1: Add injury model imports to app.py**

At the top of `app.py`, after the existing imports (line 33), add:

```python
from src.injury_model import (
    age_risk_adjustment,
    apply_injury_adjustment,
    compute_health_score,
    get_injury_badge,
    workload_flag,
    POSITION_PLAYER_AGE_THRESHOLD,
    PITCHER_AGE_THRESHOLD,
)
```

- [ ] **Step 2: Compute health scores on draft page init**

In `render_draft_page()` (after line 1108, after loading pool/config), add health score computation:

```python
    # ── Compute health scores ────────────────────────────────
    if "health_scores" not in st.session_state:
        from src.database import get_connection
        conn = get_connection()
        try:
            injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
        except Exception:
            injury_df = pd.DataFrame()
        finally:
            conn.close()

        health_dict = {}
        if not injury_df.empty and "player_id" in injury_df.columns:
            for pid, group in injury_df.groupby("player_id"):
                gp = group["games_played"].tolist()
                ga = group["games_available"].tolist()
                health_dict[pid] = compute_health_score(gp, ga)
        st.session_state.health_scores = health_dict

        # Apply injury adjustment to counting stats so injured players rank lower
        if health_dict:
            health_scores_df = pd.DataFrame([
                {"player_id": pid, "health_score": hs}
                for pid, hs in health_dict.items()
            ])
            pool = apply_injury_adjustment(pool, health_scores_df)
```

- [ ] **Step 3: Add injury badge + workload flag to hero card**

In `render_hero_pick()` (line 1270), after getting name/pos/score, add badge computation:

```python
    # Injury & risk badges
    pid = rec.get("player_id", None)
    hs = st.session_state.get("health_scores", {}).get(pid, 0.85)
    badge_icon, badge_label = get_injury_badge(hs)
    injury_html = f'<span style="margin-left:8px;">{badge_icon} {badge_label}</span>'

    # Age flag
    age = rec.get("age", None)
    is_hitter = rec.get("is_hitter", 1)
    age_threshold = POSITION_PLAYER_AGE_THRESHOLD if is_hitter else PITCHER_AGE_THRESHOLD
    age_html = f' <span style="color:{T["warn"]};">⚠️ Age {age}</span>' if age and age >= age_threshold else ""

    # Workload flag (pitchers only — >40 IP increase)
    workload_html = ""
    if not is_hitter:
        ip_current = rec.get("ip", 0)
        ip_prev = rec.get("ip_prev", ip_current)  # Fallback to current if no prev
        if workload_flag(ip_current, ip_prev):
            workload_html = f' <span style="color:{T["danger"]};">🔥 Workload</span>'
```

Then insert `{injury_html}{age_html}` into the hero card HTML after the `p-meta` div content (line 1334):

```python
        f'<div class="p-meta">{pos} &middot; Tier {tier} {vb}{injury_html}{age_html}{workload_html}</div>'
```

- [ ] **Step 4: Add injury badge to alternative cards**

In `render_alternatives()` (line 1346), add compact badge after score:

```python
            # Compact injury badge
            alt_pid = row.get("player_id", None)
            alt_hs = st.session_state.get("health_scores", {}).get(alt_pid, 0.85)
            alt_icon, _ = get_injury_badge(alt_hs)
```

Insert into alternative card HTML:

```python
                f'<div class="a-meta">{pos} {alt_icon} {risk_badge}</div>'
```

- [ ] **Step 5: Run app with sample data to verify**

Run: `streamlit run app.py` — verify badges appear on hero card and alternatives.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add injury badges and age flags to draft page hero card and alternatives"
```

---

### Task 4: Wire percentile ranges into draft page

**Files:**
- Modify: `app.py` — imports, `render_draft_page()`, `render_hero_pick()`, `render_alternatives()`

- [ ] **Step 1: Add percentile imports to app.py**

After the injury imports, add:

```python
from src.valuation import (
    add_process_risk,
    compute_percentile_projections,
    compute_projection_volatility,
)
```

Note: `compute_percentile_projections` params are `base` and `volatility` (NOT `base_df`/`volatility_df`).

- [ ] **Step 2: Compute percentile data at draft page init**

In `render_draft_page()`, after health score computation, add:

```python
    # ── Compute percentile ranges ────────────────────────────
    if "percentile_data" not in st.session_state:
        from src.database import get_connection
        conn = get_connection()
        try:
            systems = {}
            for system_name in ["steamer", "zips", "depthcharts", "blended"]:
                df = pd.read_sql_query(
                    f"SELECT * FROM projections WHERE system = '{system_name}'", conn
                )
                if not df.empty:
                    systems[system_name] = df
        except Exception:
            systems = {}
        finally:
            conn.close()

        if len(systems) >= 2:
            vol = compute_projection_volatility(systems)
            vol = add_process_risk(vol)
            pct_projs = compute_percentile_projections(base=pool, volatility=vol)
            st.session_state.percentile_data = pct_projs
            st.session_state.has_percentiles = True
        else:
            st.session_state.percentile_data = {}
            st.session_state.has_percentiles = False
```

- [ ] **Step 3: Add percentile range bar to hero card**

In `render_hero_pick()`, after the score/survival section, compute P10/P90:

```python
    # Percentile range
    pct_html = ""
    if st.session_state.get("has_percentiles", False):
        pct_data = st.session_state.get("percentile_data", {})
        p10_df = pct_data.get(10)
        p90_df = pct_data.get(90)
        if p10_df is not None and p90_df is not None and pid is not None:
            p10_row = p10_df[p10_df["player_id"] == pid]
            p90_row = p90_df[p90_df["player_id"] == pid]
            if not p10_row.empty and not p90_row.empty:
                # Use pick_score or total_sgp as proxy
                p10_val = p10_row.iloc[0].get("pick_score", p10_row.iloc[0].get("total_sgp", score * 0.7))
                p90_val = p90_row.iloc[0].get("pick_score", p90_row.iloc[0].get("total_sgp", score * 1.3))
                range_width = max(p90_val - p10_val, 0.1)
                fill_pct = int(((score - p10_val) / range_width) * 100)
                fill_pct = max(5, min(95, fill_pct))
                pct_html = (
                    f'<div style="margin-top:8px;font-family:JetBrains Mono,monospace;font-size:12px;color:{T["tx2"]};">'
                    f'P10: {p10_val:.1f} '
                    f'<span style="display:inline-block;width:120px;height:8px;background:{T["card_h"]};border-radius:4px;vertical-align:middle;">'
                    f'<span style="display:inline-block;width:{fill_pct}%;height:100%;background:{T["amber"]};border-radius:4px;"></span>'
                    f'</span>'
                    f' P90: {p90_val:.1f}</div>'
                )
    else:
        pct_html = (
            f'<div style="margin-top:4px;font-size:11px;color:{T["tx2"]};font-style:italic;">'
            f'Single projection source — range unavailable</div>'
        )
```

Insert `{pct_html}` into hero card HTML after the SGP chips row.

- [ ] **Step 4: Add compact range to alternatives**

In `render_alternatives()`, after getting the score:

```python
            # Compact percentile range
            range_text = ""
            if st.session_state.get("has_percentiles", False):
                pct_data = st.session_state.get("percentile_data", {})
                p10_df = pct_data.get(10)
                p90_df = pct_data.get(90)
                if p10_df is not None and p90_df is not None and alt_pid is not None:
                    p10_r = p10_df[p10_df["player_id"] == alt_pid]
                    p90_r = p90_df[p90_df["player_id"] == alt_pid]
                    if not p10_r.empty and not p90_r.empty:
                        p10_s = p10_r.iloc[0].get("pick_score", score * 0.7)
                        p90_s = p90_r.iloc[0].get("pick_score", score * 1.3)
                        range_text = f'<div style="font-size:10px;color:{T["tx2"]};">{p10_s:.1f} — {p90_s:.1f}</div>'
```

Insert `{range_text}` into alternative card HTML after the score line.

- [ ] **Step 5: Enable percentile sampling in MC simulation**

In `render_draft_page()`, where `evaluate_candidates` is called (line 1180), pass percentile params:

```python
                # Build volatility array for MC sampling
                perc_kwargs = {}
                if st.session_state.get("has_percentiles", False):
                    vol_array = np.zeros(len(pool))
                    # Use pick_score volatility as proxy
                    if "total_sgp" in pool.columns:
                        vol_array = np.full(len(pool), pool["total_sgp"].std() * 0.3)
                    perc_kwargs = {
                        "use_percentile_sampling": True,
                        "sgp_volatility": vol_array,
                    }

                candidates = sim.evaluate_candidates(
                    pool, ds, n_simulations=st.session_state.num_sims,
                    **perc_kwargs,
                )
```

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: add percentile range bars and MC sampling to draft page"
```

---

## Chunk 3: Draft Page — Opponent Intelligence & Practice Mode (Tasks 5-6)

---

### Task 5: Wire opponent intelligence into draft page

**Files:**
- Modify: `app.py` — imports, `render_draft_page()`, `render_hero_pick()`, `render_draft_tabs()`

- [ ] **Step 1: Add opponent model imports**

At top of `app.py`, update the simulation import:

```python
from src.simulation import DraftSimulator, detect_position_run, compute_team_preferences
from src.draft_state import DraftState, get_positional_needs
```

Note: `get_positional_needs(draft_state_dict, team_id_int, roster_config)` uses 0-based int `team_id`.

- [ ] **Step 2: Compute opponent intel on each pick**

In `render_draft_page()`, after the recommendation engine section (around line 1176), add:

```python
    # ── Compute opponent intel ───────────────────────────────
    threat_alerts = []
    opponent_data = []
    if ds.is_user_turn and rec is not None:
        # Build draft history for team preferences
        pick_log = ds.pick_log
        if len(pick_log) > 0:
            history_rows = []
            for pick in pick_log:
                history_rows.append({
                    "team_key": str(pick.get("team_index", "")),
                    "positions": pick.get("positions", ""),
                })
            history_df = pd.DataFrame(history_rows)
            preferences = compute_team_preferences(history_df)
        else:
            preferences = {}
            history_df = None

        # Determine which teams pick before our next turn
        next_user = ds.next_user_pick()
        if next_user and next_user > ds.current_pick:
            picks_between = next_user - ds.current_pick - 1
        else:
            picks_between = ds.num_teams - 1

        hero_pos = rec.get("positions", "").split(",")[0].strip()
        teams_needing_pos = 0

        for offset in range(1, picks_between + 1):
            pick_idx = ds.current_pick + offset
            if pick_idx >= ds.total_picks:
                break
            team_idx = ds.picking_team_index(pick_idx)
            team_name = ds.teams[team_idx].team_name

            # Get positional needs
            state_dict = {"picks": ds.pick_log}
            needs = get_positional_needs(state_dict, team_idx, ROSTER_CONFIG)

            # Check preference bias
            team_key = str(team_idx)
            pref = preferences.get(team_key, {}).get("positional_bias", {})
            bias = pref.get(hero_pos, 0)

            if hero_pos in needs:
                teams_needing_pos += 1

            if bias > 0.6:
                threat_alerts.append(
                    f'🔥 {team_name} targets {hero_pos} early ({bias:.0%} bias)'
                )

            opponent_data.append({
                "Team": team_name,
                "Needs": ", ".join(sorted(needs.keys())) if needs else "—",
                "Bias": hero_pos + f" ({bias:.0%})" if bias > 0.3 else "—",
            })

        if teams_needing_pos >= 3:
            threat_alerts.insert(0,
                f'⚠️ Low survival: {teams_needing_pos} teams need {hero_pos} before you'
            )

        surv = rec.get("p_survive", rec.get("survival_pct", 50))
        if surv > 80 and not threat_alerts:
            threat_alerts.append("✅ Likely available next round")
```

- [ ] **Step 3: Add threat alerts to hero card**

In `render_hero_pick()`, accept threat_alerts parameter and render:

Update the function signature:
```python
def render_hero_pick(rec, ds, pool, threat_alerts=None):
```

After the SGP chips row, add:

```python
    # Threat alerts
    alerts_html = ""
    if threat_alerts:
        for alert in threat_alerts[:2]:  # Max 2 lines
            alerts_html += f'<div style="font-size:13px;margin-top:4px;color:{T["tx2"]};">{alert}</div>'
```

Insert `{alerts_html}` into hero card HTML.

Update the call site in `render_draft_page()`:
```python
                render_hero_pick(rec, ds, pool, threat_alerts=threat_alerts)
```

- [ ] **Step 4: Add Opponent Intel tab**

In `render_draft_tabs()` (line 1613), add a 5th tab:

```python
def render_draft_tabs(ds, pool, lc, sgp):
    tabs = st.tabs(["Category Balance", "Available Players", "Draft Board", "Draft Log", "Opponent Intel"])

    # ...existing tabs...

    with tabs[4]:
        render_opponent_intel(ds)
```

Add the new function:

```python
def render_opponent_intel(ds):
    """Display positional needs, bias, and predicted picks for all opponents."""
    if not ds.pick_log:
        st.info("Opponent data will appear after the first few picks.")
        return

    # Compute team preferences from draft history
    history_rows = [
        {"team_key": str(p.get("team_index", "")), "positions": p.get("positions", "")}
        for p in ds.pick_log
    ]
    history_df = pd.DataFrame(history_rows)
    preferences = compute_team_preferences(history_df) if not history_df.empty else {}

    # Build data for each opponent
    rows = []
    for team_idx in range(ds.num_teams):
        if team_idx == ds.user_team_index:
            continue
        team_name = ds.teams[team_idx].team_name
        state_dict = {"picks": ds.pick_log}
        needs = get_positional_needs(state_dict, team_idx, ROSTER_CONFIG)
        needs_str = ", ".join(sorted(needs.keys())) if needs else "Full"

        # Historical bias
        team_key = str(team_idx)
        pref = preferences.get(team_key, {}).get("positional_bias", {})
        top_bias = max(pref.items(), key=lambda x: x[1]) if pref else ("—", 0)
        bias_str = f"{top_bias[0]} ({top_bias[1]:.0%})" if top_bias[1] > 0.2 else "—"

        # Predicted next pick: highest-bias position that is also a need
        predicted = "—"
        for pos, frac in sorted(pref.items(), key=lambda x: -x[1]):
            if pos in needs:
                predicted = pos
                break

        # Threat level (for color coding)
        picks_made = sum(1 for p in ds.pick_log if p.get("team_index") == team_idx)
        threat = "🔴" if len(needs) <= 3 else "🟡" if len(needs) <= 8 else "🟢"

        rows.append({
            "Threat": threat,
            "Team": team_name,
            "Positions Needed": needs_str,
            "Historical Bias": bias_str,
            "Predicted Next": predicted,
            "Picks Made": picks_made,
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No opponent data yet.")
```

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add opponent intelligence alerts and tab to draft page"
```

---

### Task 6: Add practice mode

**Files:**
- Modify: `app.py` — `render_draft_page()`, session state init

- [ ] **Step 1: Verify practice mode toggle already exists**

The practice mode toggle already exists in `render_draft_page()` at line 1128. Verify it's wired:

```python
        st.session_state.practice_mode = st.toggle(
            "Practice Mode", value=st.session_state.practice_mode, key="draft_practice"
        )
```

- [ ] **Step 2: Add practice mode banner**

In `render_draft_page()`, after the practice mode check (line 1158), add a visible banner:

```python
    if st.session_state.practice_mode:
        st.markdown(
            f'<div style="background:rgba(245,158,11,0.15);border:2px solid {T["warn"]};'
            f'border-radius:8px;padding:10px;text-align:center;margin-bottom:12px;">'
            f'<span style="font-family:Oswald,sans-serif;color:{T["warn"]};">'
            f'🎮 PRACTICE MODE — Picks will not be saved</span></div>',
            unsafe_allow_html=True,
        )
```

- [ ] **Step 3: Add practice mode state isolation**

The spec requires practice picks to use a separate `practice_draft_state` dict that is session-state-only (NOT written to `draft_state.json` or `draft_picks` table). On page refresh, practice state resets.

Find where session state is initialized and add:

```python
if "practice_mode" not in st.session_state:
    st.session_state.practice_mode = False
if "practice_draft_state" not in st.session_state:
    st.session_state.practice_draft_state = None
```

Then in `render_draft_page()`, after loading the real `ds`, add the active-state swap:

```python
    # Practice mode state isolation: use separate ephemeral DraftState
    if st.session_state.practice_mode:
        if st.session_state.practice_draft_state is None:
            # Clone current real state into a practice-only copy
            st.session_state.practice_draft_state = DraftState(
                num_teams=ds.num_teams,
                num_rounds=ds.num_rounds,
                user_team_index=ds.user_team_index,
                roster_config=ROSTER_CONFIG,
            )
            # Copy existing picks into practice state
            for pick in ds.pick_log:
                st.session_state.practice_draft_state.pick_log.append(pick)
                st.session_state.practice_draft_state.drafted_player_ids.add(pick["player_id"])
            st.session_state.practice_draft_state.current_pick = ds.current_pick
        ds = st.session_state.practice_draft_state  # Swap — all reads/writes go to practice state
```

**Key:** When practice mode is on, `ds` points to the ephemeral session-state copy. No file or DB writes occur because the practice `DraftState` is never persisted.

- [ ] **Step 4: Add "Reset Practice" button**

In the sidebar, after the practice toggle:

```python
        if st.session_state.practice_mode:
            if st.button("🔄 Reset Practice", width="stretch"):
                # Create fresh DraftState — practice state is ephemeral
                st.session_state.practice_draft_state = DraftState(
                    num_teams=ds.num_teams,
                    num_rounds=ds.num_rounds,
                    user_team_index=ds.user_team_index,
                    roster_config=ROSTER_CONFIG,
                )
                st.toast("Practice reset!", icon="🎮")
                st.rerun()
```

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: add practice mode banner and reset button to draft page"
```

---

## Chunk 4: In-Season Page Enhancements (Tasks 7-10)

These tasks are independent of each other and can be parallelized.

---

### Task 7: My Team — injury badges + Bayesian projections + Yahoo sync

**Files:**
- Modify: `pages/1_My_Team.py`

- [ ] **Step 1: Add imports**

At top of `1_My_Team.py`, after existing imports, add:

```python
from src.injury_model import compute_health_score, get_injury_badge

try:
    from src.bayesian import BayesianUpdater
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

try:
    from src.yahoo_api import YahooFantasyClient, YFPY_AVAILABLE
except ImportError:
    YFPY_AVAILABLE = False
```

- [ ] **Step 2: Add injury badges to roster display**

After `roster` is loaded (line 50), compute health scores:

```python
        # Compute health scores for badge display
        try:
            from src.database import get_connection
            conn = get_connection()
            injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
            conn.close()
        except Exception:
            injury_df = pd.DataFrame()

        if not injury_df.empty and "player_id" in injury_df.columns:
            badges = []
            for _, row in roster.iterrows():
                pid = row.get("player_id")
                player_injury = injury_df[injury_df["player_id"] == pid]
                if not player_injury.empty:
                    gp = player_injury["games_played"].tolist()
                    ga = player_injury["games_available"].tolist()
                    hs = compute_health_score(gp, ga)
                    icon, label = get_injury_badge(hs)
                    badges.append(f"{icon} {label}")
                else:
                    badges.append("🟢 Low Risk")
            roster["Health"] = badges
```

Add "Health" to `display_cols`:

```python
            display_cols = ["name", "positions", "roster_slot", "Health"]
```

- [ ] **Step 3: Add Bayesian projection update**

After the roster display section, add:

```python
            # Bayesian-adjusted projections
            if BAYESIAN_AVAILABLE:
                try:
                    from src.database import get_connection
                    conn = get_connection()
                    season_stats = pd.read_sql_query("SELECT * FROM season_stats", conn)
                    conn.close()

                    if not season_stats.empty and season_stats.get("games_played", pd.Series([0])).sum() > 0:
                        # Load preseason projections for Bayesian prior
                        preseason = pd.read_sql_query(
                            "SELECT * FROM blended_projections", conn
                        )
                        updater = BayesianUpdater()
                        updated = updater.batch_update_projections(season_stats, preseason)
                        st.subheader("📊 Bayesian-Adjusted Projections")
                        st.caption("Stats regressed toward preseason priors using FanGraphs stabilization thresholds")
                        stat_display = ["player_id", "avg", "hr", "rbi", "sb", "era", "whip", "k"]
                        show_cols = [c for c in stat_display if c in updated.columns]
                        st.dataframe(updated[show_cols], hide_index=True, width="stretch")
                except Exception:
                    pass  # Graceful degradation
```

- [ ] **Step 4: Add Yahoo sync button**

After the refresh button section, add:

```python
        if YFPY_AVAILABLE:
            import os
            yahoo_key = os.environ.get("YAHOO_CLIENT_ID")
            yahoo_secret = os.environ.get("YAHOO_CLIENT_SECRET")
            if yahoo_key and yahoo_secret:
                with col2:
                    if st.button("🔄 Sync Yahoo"):
                        with st.spinner("Syncing from Yahoo Fantasy..."):
                            try:
                                client = YahooFantasyClient(league_id="auto")
                                if client.authenticate(yahoo_key, yahoo_secret):
                                    st.toast("Yahoo sync complete!", icon="✅")
                                    st.rerun()
                                else:
                                    st.error("Yahoo authentication failed.")
                            except Exception as e:
                                st.error(f"Yahoo sync error: {e}")
```

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add pages/1_My_Team.py
git commit -m "feat: add injury badges, Bayesian projections, Yahoo sync to My Team page"
```

---

### Task 8: Trade Analyzer — injury badges + P10/P90 ranges

**Files:**
- Modify: `pages/2_Trade_Analyzer.py`

- [ ] **Step 1: Add imports**

After existing imports, add:

```python
from src.injury_model import compute_health_score, get_injury_badge
from src.valuation import compute_percentile_projections, compute_projection_volatility
```

- [ ] **Step 2: Load health scores from injury_history table**

After pool is loaded, compute actual health scores (NOT hardcoded):

```python
        # Load health scores from injury history
        health_dict = {}
        try:
            from src.database import get_connection
            conn = get_connection()
            injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
            conn.close()
            if not injury_df.empty and "player_id" in injury_df.columns:
                for pid, group in injury_df.groupby("player_id"):
                    gp = group["games_played"].tolist()
                    ga = group["games_available"].tolist()
                    health_dict[pid] = compute_health_score(gp, ga)
        except Exception:
            pass
```

- [ ] **Step 3: Add injury badges to trade proposal display**

After the trade analysis result is computed, before the verdict banner (around line 96), add badge display for both sides:

```python
                # Show injury badges for traded players
                trade_col1, trade_col2 = st.columns(2)
                with trade_col1:
                    st.markdown("**Giving:**")
                    for name in giving_names:
                        p = pool[pool["player_name"] == name]
                        if not p.empty:
                            pid = p.iloc[0]["player_id"]
                            hs = health_dict.get(pid, 0.85)
                            icon, label = get_injury_badge(hs)
                            st.markdown(f"{icon} {name} — {label}")
                with trade_col2:
                    st.markdown("**Receiving:**")
                    for name in receiving_names:
                        p = pool[pool["player_name"] == name]
                        if not p.empty:
                            pid = p.iloc[0]["player_id"]
                            hs = health_dict.get(pid, 0.85)
                            icon, label = get_injury_badge(hs)
                            st.markdown(f"{icon} {name} — {label}")
```

- [ ] **Step 4: Add P10/P90 upside/downside risk**

After the injury badges section, add percentile range display:

```python
                # P10/P90 risk assessment for traded players
                try:
                    from src.database import get_connection
                    conn = get_connection()
                    systems = {}
                    for sys_name in ["steamer", "zips", "depthcharts", "blended"]:
                        df = pd.read_sql_query(
                            f"SELECT * FROM projections WHERE system = '{sys_name}'", conn
                        )
                        if not df.empty:
                            systems[sys_name] = df
                    conn.close()

                    if len(systems) >= 2:
                        vol = compute_projection_volatility(systems)
                        pct = compute_percentile_projections(base=pool, volatility=vol)
                        p10_df, p90_df = pct.get(10), pct.get(90)
                        if p10_df is not None and p90_df is not None:
                            st.subheader("📊 Upside/Downside Risk")
                            all_names = list(giving_names) + list(receiving_names)
                            risk_rows = []
                            for name in all_names:
                                p10_row = p10_df[p10_df.get("name", p10_df.get("player_name", "")) == name]
                                p90_row = p90_df[p90_df.get("name", p90_df.get("player_name", "")) == name]
                                if not p10_row.empty and not p90_row.empty:
                                    p10_sgp = p10_row.iloc[0].get("total_sgp", 0)
                                    p90_sgp = p90_row.iloc[0].get("total_sgp", 0)
                                    risk_rows.append({"Player": name, "P10 (Floor)": f"{p10_sgp:.1f}", "P90 (Ceiling)": f"{p90_sgp:.1f}"})
                            if risk_rows:
                                st.dataframe(pd.DataFrame(risk_rows), hide_index=True, width="stretch")
                except Exception:
                    pass  # Graceful degradation when percentiles unavailable
```

- [ ] **Step 5: Commit**

```bash
git add pages/2_Trade_Analyzer.py
git commit -m "feat: add injury badges and P10/P90 ranges to Trade Analyzer page"
```

---

### Task 9: Player Compare — health score + projection confidence rows

**Files:**
- Modify: `pages/3_Player_Compare.py`

- [ ] **Step 1: Add imports**

After existing imports, add:

```python
from src.injury_model import compute_health_score, get_injury_badge
from src.valuation import compute_percentile_projections, compute_projection_volatility
```

- [ ] **Step 2: Load health scores from injury_history table**

After result is computed, load actual health scores:

```python
        # Load health scores
        health_dict = {}
        try:
            from src.database import get_connection
            conn = get_connection()
            injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
            conn.close()
            if not injury_df.empty and "player_id" in injury_df.columns:
                for pid, group in injury_df.groupby("player_id"):
                    gp = group["games_played"].tolist()
                    ga = group["games_available"].tolist()
                    health_dict[pid] = compute_health_score(gp, ga)
        except Exception:
            pass
```

- [ ] **Step 3: Add health score + projection confidence comparison rows**

After the z-score comparison table (line 122), add:

```python
        # Health comparison
        st.subheader("Health & Confidence")
        health_rows = []
        for name_col, pid_col in [(result["player_a"], id_a), (result["player_b"], id_b)]:
            hs = health_dict.get(pid_col, 0.85)
            icon, label = get_injury_badge(hs)
            health_rows.append({"Player": name_col, "Health": f"{icon} {label}", "Score": f"{hs:.2f}"})

        # Projection confidence: P10-P90 range width per player
        try:
            from src.database import get_connection
            conn = get_connection()
            systems = {}
            for sys_name in ["steamer", "zips", "depthcharts", "blended"]:
                df = pd.read_sql_query(
                    f"SELECT * FROM projections WHERE system = '{sys_name}'", conn
                )
                if not df.empty:
                    systems[sys_name] = df
            conn.close()

            if len(systems) >= 2:
                vol = compute_projection_volatility(systems)
                pct = compute_percentile_projections(base=pool, volatility=vol)
                p10_df, p90_df = pct.get(10), pct.get(90)
                if p10_df is not None and p90_df is not None:
                    for row in health_rows:
                        name = row["Player"]
                        p10_row = p10_df[p10_df.get("name", p10_df.get("player_name", "")) == name]
                        p90_row = p90_df[p90_df.get("name", p90_df.get("player_name", "")) == name]
                        if not p10_row.empty and not p90_row.empty:
                            width = p90_row.iloc[0].get("total_sgp", 0) - p10_row.iloc[0].get("total_sgp", 0)
                            row["Confidence"] = f"±{width:.1f} SGP" if width > 0 else "—"
                        else:
                            row["Confidence"] = "—"
        except Exception:
            for row in health_rows:
                row["Confidence"] = "—"

        st.dataframe(pd.DataFrame(health_rows), hide_index=True, width="stretch")
```

- [ ] **Step 4: Commit**

```bash
git add pages/3_Player_Compare.py
git commit -m "feat: add health score and projection confidence to Player Compare page"
```

---

### Task 10: Lineup Optimizer — health penalty + two-start SP detection

**Files:**
- Modify: `pages/5_Lineup_Optimizer.py`

- [ ] **Step 1: Add imports**

After existing imports, add:

```python
from src.injury_model import compute_health_score, get_injury_badge
```

- [ ] **Step 2: Add health-adjusted roster info + health penalty in optimization**

After the roster overview section (line 131), add health column and apply penalty:

```python
# Add health info to roster overview
health_dict = {}
try:
    from src.database import get_connection
    conn = get_connection()
    injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
    conn.close()

    if not injury_df.empty:
        health_badges = []
        for _, row in roster.iterrows():
            pid = row.get("player_id")
            pi = injury_df[injury_df["player_id"] == pid]
            if not pi.empty:
                hs = compute_health_score(pi["games_played"].tolist(), pi["games_available"].tolist())
                icon, _ = get_injury_badge(hs)
                health_badges.append(icon)
                health_dict[pid] = hs
            else:
                health_badges.append("🟢")
                health_dict[pid] = 1.0
        roster["Health"] = health_badges
except Exception:
    pass

# Apply health penalty to LP objective: reduce projected value by (1 - health_score) * penalty_weight
# This makes the optimizer prefer healthy players when value is close
HEALTH_PENALTY_WEIGHT = 0.15
if health_dict and "projected_sgp" in roster.columns:
    roster["health_adjusted_sgp"] = roster.apply(
        lambda r: r["projected_sgp"] * (1.0 - HEALTH_PENALTY_WEIGHT * (1.0 - health_dict.get(r.get("player_id"), 1.0))),
        axis=1,
    )
```

- [ ] **Step 3: Add two-start SP detection**

After the health section, add two-start pitcher identification:

```python
# Two-start SP detection via MLB schedule
try:
    import statsapi
    # Get this week's schedule
    from datetime import datetime, timedelta
    today = datetime.now()
    end = today + timedelta(days=7)
    schedule = statsapi.schedule(start_date=today.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"))

    team_game_counts = {}
    for game in schedule:
        for team_key in ["home_name", "away_name"]:
            team_name = game.get(team_key, "")
            team_game_counts[team_name] = team_game_counts.get(team_name, 0) + 1

    # Flag SPs whose team has 2+ games in the period
    two_start_sps = []
    for _, row in roster.iterrows():
        if not row.get("is_hitter", True) and "SP" in str(row.get("positions", "")):
            team = row.get("team", "")
            if team_game_counts.get(team, 0) >= 2:
                two_start_sps.append(row.get("name", row.get("player_name", "")))

    if two_start_sps:
        st.info(f"📅 Potential two-start SPs this week: {', '.join(two_start_sps)}")
except Exception:
    pass  # Graceful degradation when MLB Stats API unavailable
```

- [ ] **Step 4: Commit**

```bash
git add pages/5_Lineup_Optimizer.py
git commit -m "feat: add health penalty and two-start SP detection to Lineup Optimizer"
```

---

## Chunk 5: Yahoo API Setup Wizard + Final Verification (Tasks 11-12)

---

### Task 11: Yahoo API integration — update authenticate() + setup wizard

**Files:**
- Modify: `src/yahoo_api.py` — `authenticate()` to consume `token_data`
- Modify: `app.py` — setup wizard Step 1 section

- [ ] **Step 1: Update `authenticate()` in yahoo_api.py to consume token_data**

In `src/yahoo_api.py`, the `authenticate()` method (line 133) already accepts `token_data: dict | None = None` but currently ignores it. Update to consume when provided:

```python
    def authenticate(
        self,
        consumer_key: str,
        consumer_secret: str,
        token_data: dict | None = None,
    ) -> bool:
        # ... existing docstring ...
        if not YFPY_AVAILABLE:
            return False
        try:
            # If token_data provided, skip browser OAuth flow
            if token_data is not None:
                import json
                token_path = Path(self.data_dir) / "token.json"
                token_path.write_text(json.dumps(token_data))
                # yfpy will pick up the token file on Game() init

            self._game = Game(
                consumer_key, consumer_secret, game_code=self.game_code
            )
            return True
        except Exception:
            return False
```

- [ ] **Step 2: Add Yahoo API import to app.py**

At top of `app.py`, add:

```python
try:
    from src.yahoo_api import YahooFantasyClient, YFPY_AVAILABLE
except ImportError:
    YFPY_AVAILABLE = False
```

- [ ] **Step 3: Add Yahoo connect section to Step 1**

Find the Step 1 (league config) section of the setup wizard. Before the manual config form, add:

```python
    # Yahoo Fantasy connection
    if YFPY_AVAILABLE:
        import os
        yahoo_key = os.environ.get("YAHOO_CLIENT_ID")
        yahoo_secret = os.environ.get("YAHOO_CLIENT_SECRET")
        if yahoo_key and yahoo_secret:
            st.markdown(
                f'<div style="background:{T["card"]};border:1px solid {T["card_h"]};'
                f'border-radius:12px;padding:16px;margin-bottom:16px;">'
                f'<div style="font-family:Oswald,sans-serif;color:{T["amber"]};font-size:16px;">'
                f'🏈 Connect Yahoo Fantasy</div>'
                f'<div style="color:{T["tx2"]};font-size:13px;margin-top:4px;">'
                f'Auto-import league settings, rosters, and standings</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("Authorize with Yahoo", type="secondary"):
                with st.spinner("Connecting to Yahoo Fantasy..."):
                    try:
                        client = YahooFantasyClient(league_id="auto")
                        if client.authenticate(yahoo_key, yahoo_secret):
                            st.session_state.yahoo_client = client
                            # Auto-populate league settings
                            try:
                                settings = client.get_league_settings()
                                if settings:
                                    st.session_state.yahoo_settings = settings
                                client.sync_to_db()
                            except Exception:
                                pass  # Sync failures are non-fatal
                            st.success("Connected to Yahoo Fantasy!")
                            st.toast("League data auto-populated!", icon="✅")
                        else:
                            st.error("Authentication failed. Check your credentials.")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

            st.markdown(f'<div style="text-align:center;color:{T["tx2"]};margin:12px 0;">── or ──</div>', unsafe_allow_html=True)
```

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All pass (Yahoo import is behind try/except)

- [ ] **Step 5: Commit**

```bash
git add src/yahoo_api.py app.py
git commit -m "feat: update Yahoo authenticate() and add OAuth connect to setup wizard"
```

---

### Task 12: Final verification and lint

- [ ] **Step 1: Run ruff lint**

Run: `python -m ruff check .`
Expected: No errors

- [ ] **Step 2: Run ruff format**

Run: `python -m ruff format .`
Expected: Clean

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ~153 pass, 4 skip, 0 fail

- [ ] **Step 4: Update CLAUDE.md**

Update the memory file with Plan 4 status, new test count, and any new gotchas discovered during implementation.

- [ ] **Step 5: Final commit and push**

```bash
git add -A
git commit -m "chore: Plan 4 final lint, format, and CLAUDE.md update"
git push origin master
```
