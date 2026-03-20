# Player Card Dialog — Design Spec

> Interactive modal card showing full player profile, stats, projections, and analysis.
> Triggered from any page by selecting a player name.

---

## Overview

When a user selects a player name on any page, a full-screen modal dialog opens showing a deep-dive player card: headshot, bio, 3-year historical stats, 6-system projection breakdown, radar chart vs league/MLB averages, advanced metrics, injury timeline, rankings, news, and prospect scouting grades.

## Architecture

### Data Layer: `src/player_card.py`

Pure function `build_player_card_data(player_id) -> dict` assembles all data from the database. No Streamlit dependency — fully unit-testable.

**Return shape:**
```python
{
    "profile": {
        "name", "team", "positions", "bats", "throws", "age",
        "headshot_url", "health_score", "health_label", "tags"
    },
    "projections": {
        "blended": {12 category stats},
        "systems": {system_name: {12 category stats}, ...}
    },
    "historical": [
        {"season", "PA"/"IP", 12 category stats, "games_played"}, ...  # newest first
    ],
    "advanced": {"FIP", "xFIP", "SIERA", "Stuff+", "Location+", "Pitching+"},  # pitchers only
    "injury_history": [
        {"season", "GP", "GA", "IL_stints", "IL_days"}, ...
    ],
    "rankings": {
        "consensus_rank", "rank_range", "yahoo_adp", "fantasypros_adp", "nfbc_adp"
    },
    "radar": {
        "player": {6 cats: 0-100 percentile},
        "league_avg": {6 cats: 0-100},
        "mlb_avg": {6 cats: 0-100}
    },
    "trends": {cat: [3 yearly values], ...},
    "news": [{"headline", "source", "date", "sentiment"}, ...],  # deduplicated, max 5
    "prospect": None or {scouting grades, MiLB stats, ETA, risk, report}
}
```

**Helper functions:**
- `_compute_radar_percentiles(player_id, position, pool_df)` — Percentile rank vs league rosters and full pool
- `_build_sparkline_data(historical)` — 3-year trend arrays per category
- `_get_headshot_url(mlb_id)` — MLB static headshot URL pattern
- `_dedup_news(news_items)` — Case-insensitive headline dedup, includes full datetime

### Rendering Layer: `src/ui_shared.py`

**`show_player_card_dialog(player_id)`** — `@st.dialog("Player Card", width="large")`

Renders 9 vertical sections inside the dialog:
1. **Header** — Headshot (80px round), name, team/pos/age, health dot, tag badges. HTML via `st.markdown`.
2. **Radar chart** — 6-axis Plotly polar chart with 3 traces (player/league avg/MLB avg).
3. **Historical stats** — 3 seasons via `render_compact_table`, newest first.
4. **Projections** — All systems + bolded blended row via `render_compact_table`.
5. **Advanced metrics** — FIP/xFIP/SIERA/Stuff+ grid. Pitchers only.
6. **Rankings & ADP** — Consensus rank, ADP from 3 sources, rank range bar.
7. **Injury history** — Season table with green/red row tinting by health.
8. **News** — Last 5 unique headlines with source badge, full datetime, sentiment dot.
9. **Prospect scouting** — FG grades, MiLB stats, ETA, risk. Conditional.

**Player selection widget** — `st.selectbox` placed below each compact table. Selecting a player calls `show_player_card_dialog(player_id)`.

### Integration: All Pages

Every page with player data gets a selectbox below its main table:
- My Team, Draft Simulator, Trade Analyzer, Player Compare, Free Agents
- Lineup Optimizer, Closer Monitor, Leaders, app.py draft page
- Not Standings (team-level data only)

### News Deduplication

Headlines are deduped case-insensitively. Each news item includes full datetime (e.g., "Mar 20, 2026 at 2:15 PM"). Max 5 items per player.

## Data Sources

| Data | Table | Key |
|------|-------|-----|
| Profile | `players` | player_id |
| Headshot | MLB static URL | mlb_id |
| Projections | `projections` | player_id, system |
| Historical | `season_stats` | player_id, season |
| Injury | `injury_history` | player_id, season |
| ADP | `adp` | player_id |
| ECR | `ecr_consensus` | player_id |
| Tags | `player_tags` | player_id |
| News | `player_news` | player_id |
| Prospects | `prospect_rankings` | mlb_id |

## Files

| File | Change | Est. Lines |
|------|--------|------------|
| `src/player_card.py` | New — data assembly | ~300 |
| `src/ui_shared.py` | Add dialog + select widget | +150 |
| `tests/test_player_card.py` | New — ~42 tests | ~350 |
| 9 page files + app.py | Add selectbox integration | +10 each |

## Constraints

- No emoji — icons via PAGE_ICONS SVGs only
- All news items unique by headline (case-insensitive)
- All news items include full datetime
- Radar chart uses HEATER color palette (orange/navy/gray)
- Health dots use existing CSS `.health-dot` class
- Tag badges use existing THEME colors
- `build_player_card_data()` must be pure — no Streamlit calls
- Headshot falls back gracefully when mlb_id is missing
