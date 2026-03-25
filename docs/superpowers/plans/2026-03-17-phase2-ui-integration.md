# Phase 2 UI Integration — Draft Recommendation Engine

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the completed DraftRecommendationEngine (5 backend modules, 270 tests) into the Streamlit UI — both app.py (main draft page) and pages/2_Draft_Simulator.py (mock draft). Display enhanced scoring, BUY/FAIR/AVOID badges, category balance meters, and mode selection.

**Architecture:** The DraftRecommendationEngine.recommend() method replaces the direct DraftSimulator.evaluate_candidates() call in both app.py and the Draft Simulator page. It internally swaps pick_score with enhanced_pick_score, runs MC simulation, and returns results with 15 extra columns. The UI displays these new columns as visual badges, meters, and tooltips within the existing glassmorphic card system. No new pages or layout changes — all enhancements are additive within existing UI components.

**Tech Stack:** Streamlit, existing THEME/T dict from ui_shared.py, inline SVG icons from PAGE_ICONS, glassmorphic CSS from inject_custom_css()

---

## File Map

### Modified Files
- `app.py` — Import DraftRecommendationEngine, initialize in session state, call engine.recommend() instead of direct evaluate_candidates(), display new columns in hero card and alternatives grid
- `pages/2_Draft_Simulator.py` — Same integration as app.py but for the standalone mock draft page
- `src/simulation.py` — Add category_weights parameter to evaluate_candidates() for category-aware MC scoring
- `src/ui_shared.py` — Add CSS classes for BUY/FAIR/AVOID badges, category balance meter, mode selector styling
- `tests/test_simulation_math.py` — Add tests for category-weighted MC scoring

### No New Files
All backend modules already exist. This plan is pure integration + display.

---

## Chunk 1: MC Enhancement + app.py Integration

### Task 1: Category-Aware MC Scoring

**Files:**
- Modify: `src/simulation.py` — `evaluate_candidates()` method
- Test: `tests/test_simulation_math.py`

- [ ] **Step 1: Write failing test for category_weights parameter**

```python
# In tests/test_simulation_math.py, add:
def test_evaluate_candidates_accepts_category_weights():
    """evaluate_candidates() accepts optional category_weights dict."""
    sim = DraftSimulator(LeagueConfig())
    pool = _make_pool(10)
    ds = _make_draft_state()
    weights = {"r": 1.5, "hr": 0.8, "rbi": 1.0, "sb": 1.2, "avg": 1.0, "obp": 1.0,
               "w": 1.0, "l": 1.0, "sv": 1.0, "k": 1.0, "era": 1.0, "whip": 1.0}
    result = sim.evaluate_candidates(pool, ds, top_n=5, category_weights=weights)
    assert len(result) <= 5
    assert "mc_mean" in result.columns or "combined_score" in result.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_simulation_math.py::test_evaluate_candidates_accepts_category_weights -v`
Expected: FAIL (unexpected keyword argument)

- [ ] **Step 3: Add category_weights parameter to evaluate_candidates()**

In `src/simulation.py`, add `category_weights=None` parameter to `evaluate_candidates()`. When provided, multiply each player's per-category SGP contribution by the corresponding weight before summing. This makes the MC simulation category-aware — weak categories get boosted.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_simulation_math.py::test_evaluate_candidates_accepts_category_weights -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/simulation.py tests/test_simulation_math.py
git commit -m "feat: add category_weights parameter to evaluate_candidates()"
```

### Task 2: Wire DraftRecommendationEngine into app.py

**Files:**
- Modify: `app.py` — imports, session state init, draft page recommendation call

- [ ] **Step 1: Add import with fallback**

At the top of `app.py`, add:
```python
try:
    from src.draft_engine import DraftRecommendationEngine
    HAS_DRAFT_ENGINE = True
except ImportError:
    HAS_DRAFT_ENGINE = False
```

- [ ] **Step 2: Initialize engine in session state during bootstrap**

After `st.session_state.bootstrap_complete = True`, add:
```python
if HAS_DRAFT_ENGINE and "draft_engine" not in st.session_state:
    from src.valuation import LeagueConfig
    config = st.session_state.get("league_config", LeagueConfig())
    mode = st.session_state.get("engine_mode", "standard")
    st.session_state.draft_engine = DraftRecommendationEngine(config, mode=mode)
```

- [ ] **Step 3: Replace evaluate_candidates() call with engine.recommend()**

Find where `simulator.evaluate_candidates()` is called on the draft page. Replace with:
```python
if HAS_DRAFT_ENGINE and "draft_engine" in st.session_state:
    engine = st.session_state.draft_engine
    recs = engine.recommend(
        player_pool=pool,
        draft_state=ds,
        top_n=8,
        n_simulations=n_sims,
        park_factors=st.session_state.get("park_factors"),
    )
    recs["player_name"] = recs["name"]  # alias for UI
else:
    # Fallback to direct MC simulation
    recs = simulator.evaluate_candidates(pool, ds, top_n=8, n_simulations=n_sims)
    recs["player_name"] = recs["name"]
```

- [ ] **Step 4: Add mode selector to Settings step**

In the setup wizard Settings step, add a radio button:
```python
engine_mode = st.radio(
    "Recommendation Engine Mode",
    ["quick", "standard", "full"],
    index=1,  # default: standard
    format_func=lambda m: {"quick": "Quick (<1s)", "standard": "Standard (2-3s)", "full": "Full (5-10s)"}[m],
    horizontal=True,
)
st.session_state["engine_mode"] = engine_mode
```

- [ ] **Step 5: Verify app launches without errors**

Run: `streamlit run app.py` — confirm splash screen completes, setup wizard shows mode selector, draft page renders with enhanced recommendations.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: wire DraftRecommendationEngine into app.py draft page"
```

### Task 3: Display BUY/FAIR/AVOID Badges

**Files:**
- Modify: `app.py` — hero card and alternatives grid rendering
- Modify: `src/ui_shared.py` — CSS for badges

- [ ] **Step 1: Add CSS for BUY/FAIR/AVOID badges**

In `src/ui_shared.py`, inside `inject_custom_css()`, add:
```css
.badge-buy { background: linear-gradient(135deg, #2d6a4f, #40916c); color: white;
    padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px; }
.badge-fair { background: linear-gradient(135deg, #457b9d, #5a9ab5); color: white;
    padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px; }
.badge-avoid { background: linear-gradient(135deg, #c1121f, #e63946); color: white;
    padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px; }
```

- [ ] **Step 2: Render badge on hero card**

In the hero card rendering section of app.py, add after the player name:
```python
bfa = rec.get("buy_fair_avoid", "fair")
badge_class = f"badge-{bfa.lower()}"
st.markdown(f'<span class="{badge_class}">{bfa.upper()}</span>', unsafe_allow_html=True)
```

- [ ] **Step 3: Render badge on alternative cards**

In the alternatives grid rendering, add the same badge to each card.

- [ ] **Step 4: Visual verification**

Run the app and verify BUY (green), FAIR (blue), and AVOID (red) badges display correctly on hero and alternative cards.

- [ ] **Step 5: Commit**

```bash
git add app.py src/ui_shared.py
git commit -m "feat: display BUY/FAIR/AVOID badges on draft recommendation cards"
```

### Task 4: Display Category Balance + Enhanced Metrics

**Files:**
- Modify: `app.py` — hero card expanded details

- [ ] **Step 1: Show category balance multiplier as a meter**

In the hero card details section, add a category need indicator:
```python
cat_mult = rec.get("category_balance_multiplier", 1.0)
need_pct = min(100, int((cat_mult - 0.8) / 0.4 * 100))  # 0.8→0%, 1.2→100%
need_color = T["green"] if need_pct < 40 else T["hot"] if need_pct < 70 else T["primary"]
st.markdown(f'''
<div style="margin: 4px 0;">
    <span style="font-size: 0.8rem; color: {T["muted"]};">Category Need</span>
    <div style="background: {T["border"]}; border-radius: 4px; height: 6px; margin-top: 2px;">
        <div style="background: {need_color}; width: {need_pct}%; height: 100%; border-radius: 4px;"></div>
    </div>
</div>''', unsafe_allow_html=True)
```

- [ ] **Step 2: Show timing info**

After recommendations render, show engine timing:
```python
if HAS_DRAFT_ENGINE and hasattr(st.session_state.get("draft_engine"), "timing"):
    timing = st.session_state.draft_engine.timing
    total = timing.get("total", 0)
    st.caption(f"Engine: {total:.1f}s ({st.session_state.get('engine_mode', 'standard')} mode)")
```

- [ ] **Step 3: Show enhanced metrics in hero card tooltip area**

Display injury probability, Statcast delta, closer bonus, and streaming penalty when available:
```python
cols = st.columns(4)
if "injury_probability" in rec:
    cols[0].metric("Injury Risk", f"{rec['injury_probability']:.0%}")
if "statcast_delta" in rec and rec["statcast_delta"] != 0:
    cols[1].metric("Skill Delta", f"{rec['statcast_delta']:+.3f}")
if "closer_hierarchy_bonus" in rec and rec["closer_hierarchy_bonus"] > 0:
    cols[2].metric("Closer Bonus", f"+{rec['closer_hierarchy_bonus']:.1f}")
if "streaming_penalty" in rec and rec["streaming_penalty"] < 0:
    cols[3].metric("Stream Penalty", f"{rec['streaming_penalty']:.1f}")
```

- [ ] **Step 4: Visual verification**

Run the app and verify category balance meter, timing, and enhanced metrics display correctly.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: display category balance meter + enhanced draft metrics on hero card"
```

---

## Chunk 2: Draft Simulator Integration + Final Polish

### Task 5: Wire DraftRecommendationEngine into Draft Simulator Page

**Files:**
- Modify: `pages/2_Draft_Simulator.py`

- [ ] **Step 1: Add import with fallback**

Same pattern as app.py — import DraftRecommendationEngine with HAS_DRAFT_ENGINE flag.

- [ ] **Step 2: Initialize engine in session state**

When mock draft starts, create engine instance:
```python
if HAS_DRAFT_ENGINE and "mock_draft_engine" not in st.session_state:
    config = st.session_state.get("mock_lc", LeagueConfig())
    st.session_state.mock_draft_engine = DraftRecommendationEngine(config, mode="standard")
```

- [ ] **Step 3: Replace evaluate_candidates() with engine.recommend()**

Same pattern as app.py Task 2 Step 3, using `mock_draft_engine` and mock draft state.

- [ ] **Step 4: Display BUY/FAIR/AVOID badges + category balance**

Reuse the same CSS classes and rendering logic from Tasks 3-4.

- [ ] **Step 5: Add mode selector**

Add Quick/Standard/Full radio button to the Draft Simulator page settings area.

- [ ] **Step 6: Verify mock draft works with enhanced engine**

Run the app, navigate to Draft Simulator, start a mock draft, and verify enhanced recommendations display correctly.

- [ ] **Step 7: Commit**

```bash
git add pages/2_Draft_Simulator.py
git commit -m "feat: wire DraftRecommendationEngine into Draft Simulator page"
```

### Task 6: Run Full Test Suite + Lint

**Files:**
- All modified files

- [ ] **Step 1: Run ruff lint**

```bash
python -m ruff check src/simulation.py app.py pages/2_Draft_Simulator.py src/ui_shared.py
```

Fix any issues.

- [ ] **Step 2: Run ruff format**

```bash
python -m ruff format src/simulation.py app.py pages/2_Draft_Simulator.py src/ui_shared.py
```

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest --tb=short -q
```

Expected: All existing tests pass, no regressions.

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "style: lint + format for Phase 2 UI integration"
```

### Task 7: Push to Master + Final Verification

- [ ] **Step 1: Push all changes**

```bash
git push
```

- [ ] **Step 2: Verify CI passes**

```bash
gh run list --limit 1
```

- [ ] **Step 3: Run the app end-to-end**

Start the app, go through the full flow:
1. Splash screen bootstrap
2. Settings step (verify mode selector)
3. Launch step
4. Draft page — verify hero card shows BUY/FAIR/AVOID badge, category balance meter, enhanced metrics
5. Draft Simulator page — verify same enhancements

- [ ] **Step 4: Update CLAUDE.md test count if changed**

If test count changed, update the `# Run all tests` comment in CLAUDE.md.
