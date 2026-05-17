# Auto-Fetch Data Pipeline — Design Spec

**Date:** 2026-03-11
**Goal:** Automatically fetch Steamer, ZiPS, and Depth Charts projections (plus ADP) from FanGraphs on app startup, eliminating manual CSV uploads.

---

## Architecture

A new module `src/data_pipeline.py` fetches projections from FanGraphs' internal JSON API on app startup when data is stale or missing. CSV upload remains as a fallback for advanced users or when the API is unavailable.

### Data Flow

```
App startup → render_step_import()
  → init_db()  (ensure tables exist)
  → check session_state["projections_fetched"]
  → not fetched yet? → refresh_if_stale()
    → projections table empty or force=True?
    → fetch_all_projections()
      → 6 HTTP GETs: 3 systems × 2 stat types
      → normalize JSON (map FG field names → DB schema)
      → for each player: upsert into players table → get player_id
      → DELETE old rows for each system → INSERT new projections
      → create_blended_projections()
      → extract ADP → resolve player_id → INSERT into adp table
      → log refresh timestamp to refresh_log
  → show status (success/partial/fail)
  → CSV uploaders available below as "Manual override"
```

### API Endpoints (Confirmed Working)

Base URL: `https://www.fangraphs.com/api/projections`

| System | Hitters | Pitchers |
|--------|---------|----------|
| Steamer | `?type=steamer&stats=bat&pos=all&team=0&lg=all&players=0` | `?type=steamer&stats=pit&pos=all&team=0&lg=all&players=0` |
| ZiPS | `?type=zips&stats=bat&pos=all&team=0&lg=all&players=0` | `?type=zips&stats=pit&pos=all&team=0&lg=all&players=0` |
| Depth Charts | `?type=fangraphsdc&stats=bat&pos=all&team=0&lg=all&players=0` | `?type=fangraphsdc&stats=pit&pos=all&team=0&lg=all&players=0` |

Each endpoint returns a JSON array of player records with 60+ fields.

### System Name Mapping

The FG API `type` parameter does not match the DB `system` column values. The pipeline must map:

```python
SYSTEM_MAP = {
    "steamer": "steamer",
    "zips": "zips",
    "fangraphsdc": "depthcharts",  # FG API name → DB system name
}
```

This is critical: the codebase (including the percentile volatility pipeline in app.py) iterates over `["steamer", "zips", "depthcharts"]` to find per-system projections. Storing `"fangraphsdc"` would break P10/P50/P90 features.

### Staleness Logic

- **"On app startup only"** — fetch once per Streamlit session
- Tracked via `st.session_state["projections_fetched"]` (boolean)
- First time Step 1 renders → auto-fetch fires
- Subsequent visits to Step 1 in same session → skip
- `force=True` parameter available for manual "Refresh" button
- `refresh_if_stale()` also checks if `projections` table is empty (first run)
- Each successful fetch writes a timestamp to `refresh_log` table (source=`"fangraphs_projections"`)

---

## Module: `src/data_pipeline.py`

### Constants

```python
_BASE_URL = "https://www.fangraphs.com/api/projections"
_TIMEOUT = 10  # seconds per request
_RATE_LIMIT = 0.5  # seconds between requests

SYSTEM_MAP = {
    "steamer": "steamer",
    "zips": "zips",
    "fangraphsdc": "depthcharts",
}

SYSTEMS = ["steamer", "zips", "fangraphsdc"]
```

### Public API

```python
def refresh_if_stale(force: bool = False) -> bool:
    """Fetch projections if not yet fetched this session.

    Calls init_db() to ensure tables exist.
    Orchestrates: fetch → normalize → upsert players → store projections → blend → ADP.
    Returns True if data was refreshed, False if already fresh or all fetches failed.
    """

def fetch_all_projections() -> dict[str, pd.DataFrame]:
    """Fetch all 6 endpoints (3 systems × bat/pit).

    Returns dict keyed by "{db_system}_{stats}" e.g. "steamer_bat", "depthcharts_pit".
    Missing systems (API failures) are omitted from dict.
    0.5s delay between requests to respect rate limits.
    """

def fetch_projections(system: str, stats: str) -> pd.DataFrame:
    """Single API call to FanGraphs projections endpoint.

    Args:
        system: FG API type — "steamer", "zips", or "fangraphsdc"
        stats: "bat" or "pit"

    Returns normalized DataFrame with columns matching projections table schema,
    plus a `name`, `team`, `positions`, `is_hitter` columns for player upsert.
    Raises FetchError on network/parse failure.
    """

def extract_adp(hitters_df: pd.DataFrame, pitchers_df: pd.DataFrame) -> pd.DataFrame:
    """Pull ADP from Steamer responses (most complete ADP data).

    Filters out ADP >= 999 and null values.
    Returns DataFrame with columns: name, adp.
    The caller (_store_adp) resolves name → player_id before DB insert.
    """
```

### Internal Helpers

```python
def normalize_hitter_json(raw: list[dict]) -> pd.DataFrame:
    """Map FanGraphs JSON hitter fields to DB schema.

    Sets is_hitter=True. Extracts primary position from 'minpos' field.
    NOTE: FG 'minpos' provides primary position only (e.g., "SS"), not
    multi-position eligibility. This is a known limitation vs CSV import.
    """

def normalize_pitcher_json(raw: list[dict]) -> pd.DataFrame:
    """Map FanGraphs JSON pitcher fields to DB schema.

    Sets is_hitter=False. Classifies SP/RP using existing logic from
    import_pitcher_csv(): GS >= 5 or IP >= 80 → "SP", SV >= 3 → "RP",
    GS >= 1 → "SP,RP", else → "RP".
    Adds 'positions' column with the classification result.
    """

def _store_projections(projections: dict[str, pd.DataFrame]) -> int:
    """Upsert players then store projections in DB.

    For each system being stored:
    1. DELETE FROM projections WHERE system = ? (idempotency)
    2. For each player row: upsert into players table via _upsert_player()
    3. INSERT projection row with resolved player_id
    Returns total row count inserted.
    """

def _store_adp(adp_df: pd.DataFrame) -> int:
    """Resolve player names to player_ids and store ADP.

    Uses exact-match against players table (name field).
    Falls back to fuzzy match for unresolved names.
    DELETE FROM adp before INSERT (idempotency).
    Returns row count.
    """

def _upsert_player(name: str, team: str, positions: str, is_hitter: bool) -> int:
    """Insert or update player in players table. Returns player_id.

    Reuses the same pattern as database.py's existing _upsert_player().
    """
```

### Column Mapping: FanGraphs JSON → DB Schema

**Hitters:**

| FG Field | DB Column | Notes |
|----------|-----------|-------|
| `PlayerName` | `name` | For player upsert |
| `Team` | `team` | 3-letter abbreviation |
| `minpos` | `positions` | Primary position only (known limitation — no multi-pos eligibility) |
| `PA` | `pa` | |
| `AB` | `ab` | |
| `H` | `h` | |
| `R` | `r` | |
| `HR` | `hr` | |
| `RBI` | `rbi` | |
| `SB` | `sb` | |
| `AVG` | `avg` | |

**Pitchers:**

| FG Field | DB Column | Notes |
|----------|-----------|-------|
| `PlayerName` | `name` | For player upsert |
| `Team` | `team` | |
| *(derived)* | `positions` | "SP", "RP", or "SP,RP" — see classification logic above |
| `IP` | `ip` | |
| `W` | `w` | |
| `SV` | `sv` | |
| `SO` | `k` | FG uses SO, DB uses k |
| `ERA` | `era` | |
| `WHIP` | `whip` | |
| `ER` | `er` | |
| `BB` | `bb_allowed` | FG uses BB for pitchers |
| `H` | `h_allowed` | Pitchers: H → h_allowed |
| `GS` | *(classification input)* | Used with G, IP, SV for SP/RP classification |
| `G` | *(classification input)* | Total games appeared |

**ADP (from Steamer response):**

| FG Field | Resolution | DB Column | Notes |
|----------|-----------|-----------|-------|
| `PlayerName` | → `player_id` via players table lookup | `player_id` | Primary key in adp table |
| `ADP` | direct | `adp` | Filter out >= 999 and null |
| *(hardcoded)* | | `source` | "fangraphs" |

### Known Limitation: Single-Position Eligibility

FanGraphs' `minpos` field only provides a player's primary defensive position (e.g., "SS"), not multi-position eligibility (e.g., "SS,2B,OF"). This means:

- Players imported via auto-fetch will have single-position eligibility
- The VORP multi-position flexibility premium (+0.12/extra position) won't apply to auto-fetched players
- **Workaround:** Users can still upload a CSV with full position eligibility to override, or a future enhancement could cross-reference MLB roster data for multi-position info
- This is an acceptable trade-off for the convenience of auto-fetch

---

## Error Handling

### Per-Request Failures
- Each of the 6 API calls has a 10-second timeout
- Failed calls log a warning but don't block other systems
- If 1 system fails: other 2 still import (partial data)
- If all 6 calls fail: show error toast, fall through to CSV upload

### Graceful Degradation Ladder
1. **All 6 succeed** → "✅ Projections loaded (Steamer + ZiPS + Depth Charts)"
2. **Some succeed** → "⚠️ Loaded {n} of 3 systems (ZiPS unavailable)"
3. **All fail** → "❌ Could not reach FanGraphs. Upload CSVs manually below."
4. **Network error** → Same as #3

### Idempotency
- `_store_projections()` DELETEs existing rows for each system before INSERTing new ones
- `_store_adp()` DELETEs all existing adp rows before INSERTing
- This ensures force-refresh and re-fetch produce clean data without duplicates
- Mirrors the pattern used by `create_blended_projections()` in database.py

### CloudFlare / Rate Limiting
- FanGraphs uses CloudFlare protection
- Add `User-Agent` header mimicking a browser
- 0.5-second delay between requests (same pattern as yahoo_api.py)
- If CloudFlare blocks: graceful fallback to CSV

---

## app.py Integration

### Step 1 Changes

In `render_step_import()`, before existing CSV uploaders:

```python
init_db()  # Ensure tables exist before any data operations

if "projections_fetched" not in st.session_state:
    with st.spinner("Fetching latest projections from FanGraphs..."):
        from src.data_pipeline import refresh_if_stale
        result = refresh_if_stale()
        st.session_state.projections_fetched = result
    if result:
        st.toast("Projections updated!", icon="✅")

# Show status card
if st.session_state.get("projections_fetched"):
    # Green card: "Projections loaded" with system list + player counts
    # Set session_state flags so Step 2+ knows data is available
    st.session_state.hitter_data = True
    st.session_state.pitcher_data = True
    # Manual refresh button (calls refresh_if_stale(force=True))
else:
    # Orange card: "Auto-fetch unavailable — upload CSVs below"
```

### CSV Fallback
Existing CSV uploaders remain below as "Manual override." Note: the existing `import_hitter_csv(hitter_file)` calls in app.py are missing the required `system` argument — this is a pre-existing bug that should be fixed as part of this work. The CSV upload should include a system selector dropdown (Steamer/ZiPS/Depth Charts) or default to `"steamer"`.

### Step 2 (League Config) Impact
No changes. League config is independent of data source.

### No Other File Changes
- `src/database.py` — no schema changes needed; reuses existing `projections`, `adp`, `players`, and `refresh_log` tables
- `src/valuation.py` — no changes; already consumes blended projections
- Pages — no changes; they read from DB regardless of how data got there

---

## Tests: `tests/test_data_pipeline.py`

~12 tests:

1. **test_normalize_hitter_json** — Sample FG response → correct column names, types, values
2. **test_normalize_hitter_position** — `minpos` field correctly maps to `positions`
3. **test_normalize_pitcher_json** — Sample FG response → correct column mapping
4. **test_normalize_pitcher_sp_classification** — GS >= 5 → "SP"
5. **test_normalize_pitcher_rp_classification** — SV >= 3 → "RP"
6. **test_normalize_pitcher_dual_classification** — GS >= 1 but < 5 → "SP,RP"
7. **test_system_name_mapping** — "fangraphsdc" stored as "depthcharts" in DB
8. **test_extract_adp_filters** — Nulls and ADP >= 999 excluded
9. **test_extract_adp_values** — Valid ADP values preserved correctly
10. **test_fetch_projections_success** — Mock HTTP → normalized DataFrame
11. **test_refresh_partial_failure** — 1 system fails, 2 succeed → returns True
12. **test_refresh_total_failure** — All systems fail → returns False
13. **test_store_idempotency** — Double-store doesn't create duplicates
14. **test_store_resolves_player_ids** — Player upsert creates valid player_id references

---

## Dependencies

- `requests` — add explicitly to `requirements.txt` (currently a transitive dep of Streamlit, but explicit is safer)
- No other new dependencies

---

## Risks

| Risk | Mitigation |
|------|-----------|
| FanGraphs changes API structure | Column mapping centralized in normalize functions; easy to update |
| CloudFlare blocks automated requests | User-Agent header, rate limiting, CSV fallback |
| API returns different fields per system | Normalize functions handle missing fields with defaults |
| ADP data incomplete | Filter unranked (999), null; users can still upload FantasyPros CSV |
| Single-position eligibility | Known limitation documented; CSV override available; future enhancement possible |
| `requests` removed from Streamlit deps | Now explicitly in requirements.txt |

---

## Success Criteria

- [ ] App startup fetches all 3 projection systems without user interaction
- [ ] System names correctly mapped (fangraphsdc → depthcharts)
- [ ] Players upserted with valid player_ids before projection INSERT
- [ ] Blended projections computed automatically from fetched data
- [ ] ADP populated from FanGraphs data with player_id resolution
- [ ] Idempotent: re-fetch produces clean data without duplicates
- [ ] Partial failures handled gracefully (2/3 systems still usable)
- [ ] Total failure falls back to CSV upload with clear messaging
- [ ] CSV upload still works as manual override (with system selector fix)
- [ ] All existing tests still pass
- [ ] ~14 new tests for the pipeline module
