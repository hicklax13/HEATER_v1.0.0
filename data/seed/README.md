# Seed Data — Tier 3 fallback for SF-7

These JSON files provide a "warm-start" baseline for the catcher framing and umpire tendencies tables when all live sources fail (FanGraphs 403, Savant umpire endpoint 404, statsapi missing data, etc.).

## When the seed fires

3-tier waterfall in `src/data_bootstrap.py`:
- **Tier 1 (primary):** pybaseball / FanGraphs / statsapi
- **Tier 2 (fallback):** direct Savant scrape with browser headers
- **Tier 3 (emergency):** seed files in this directory

When Tier 3 fires, refresh_log records `tier='emergency'` and the message includes `[SEED]`.

## Files

### catcher_framing_<YEAR>.json

```json
{
  "source": "Baseball Savant catcher-framing leaderboard",
  "scraped_date": "<YYYY-MM-DD>",
  "season": <YEAR>,
  "league_avg_framing_runs_per_game": 0.063,
  "notes": "<provenance + any caveats>",
  "catchers": [
    {
      "player_name": "Patrick Bailey",
      "team": "SF",
      "framing_runs": 14.2,
      "calls_above_avg": 158,
      "games": 124,
      "pitches_received": 9438,
      "pop_time": 1.85,
      "cs_pct": 0.282
    }
  ]
}
```

Required per-catcher fields (consumed by `_load_catcher_framing_seed()`): `player_name`, `team`, `framing_runs`, `calls_above_avg`, `games`, `pitches_received`, `pop_time`, `cs_pct`.

### umpire_tendencies_<YEAR>.json

```json
{
  "source": "Baseball Savant umpire scorecards",
  "scraped_date": "<YYYY-MM-DD>",
  "season": <YEAR>,
  "league_avg_k_pct": 0.226,
  "league_avg_bb_pct": 0.084,
  "league_avg_runs_per_game": 4.39,
  "notes": "<provenance + any caveats>",
  "umpires": [
    {
      "name": "Pat Hoberg",
      "games": 87,
      "k_pct_diff": 0.012,
      "bb_pct_diff": -0.003,
      "runs_per_game_diff": -0.18,
      "sz_consistency": 0.96
    }
  ]
}
```

Required per-umpire fields (consumed by `_load_umpire_tendencies_seed()`): `name`, `games`, `k_pct_diff`, `bb_pct_diff`, `runs_per_game_diff`, `sz_consistency`. The `_diff` values are deltas vs the `league_avg_*` fields at the top of the file.

## Annual refresh procedure

After the MLB season ends (typically late October / early November):

### Catcher framing
1. Visit `https://baseballsavant.mlb.com/leaderboard/catcher-framing?year=<YEAR>` (replace `<YEAR>` with the season just completed).
2. Filter to catchers with at least 300 framed pitches (matches the existing seed threshold).
3. Export the embedded JSON. Open browser devtools, search the page HTML for `const data = [` — Savant inlines the leaderboard payload as a JS literal.
4. Transform to the schema above. If Savant is reachable from this machine, the existing Wave 5 SF-7 helper at `src.data_bootstrap._fetch_catcher_framing_savant_scrape(year)` already returns the right shape — call it and dump the result.
5. Save the file as `data/seed/catcher_framing_<YEAR>.json`.
6. Update `src/data_bootstrap.py::_load_catcher_framing_seed()` if it has a hardcoded year in the file path (it currently points at `catcher_framing_2024.json` at line 1821).

### Umpire tendencies
Umpire data is the harder of the two: **Baseball Savant has no public umpire leaderboard endpoint**, so there is no pybaseball / scrape path that "just works". Recent seasons require manual aggregation from community sources.

1. Pull from a public umpire-scorecards source. Options that have worked historically:
   - Umpire Scorecards on Twitter/X — annual season-end recaps with per-umpire totals.
   - Community datasets shared on r/baseball or sabr.org forums.
   - Independent scrapes of the Savant pitch-by-pitch CSVs grouped by umpire (heaviest lift, highest fidelity).
2. For each active umpire collect: `games`, `k_pct_diff`, `bb_pct_diff`, `runs_per_game_diff`, `sz_consistency`. Compute `_diff` values relative to that season's league averages and store the league averages at the top of the file.
3. Save as `data/seed/umpire_tendencies_<YEAR>.json`.
4. Update `src/data_bootstrap.py::_load_umpire_tendencies_seed()` if it has a hardcoded year in the file path (it currently points at `umpire_tendencies_2024.json` at line 1483).

## Verification

After refresh, run the seed-file integrity tests:

```bash
python -m pytest tests/test_sf7_seed_files.py -v
```

Then run the bootstrap to populate the live DB from the new seed:

```python
from src.data_bootstrap import _bootstrap_catcher_framing, _bootstrap_umpire_tendencies, BootstrapProgress
_bootstrap_catcher_framing(BootstrapProgress())
_bootstrap_umpire_tendencies(BootstrapProgress())
```

Verify in the live DB:

```sql
SELECT COUNT(*) FROM catcher_framing;
SELECT COUNT(*) FROM umpire_tendencies;
SELECT status, tier, message FROM refresh_log WHERE source IN ('catcher_framing', 'umpire_tendencies');
```

Expected: row counts roughly match the new seed file size (32 catchers / 34 umpires for the current 2024 baseline). `tier='emergency'` with a `[SEED]` message indicates the live fetch failed and the seed fired — that's the correct behavior, not an error.

## History

- 2026-05-10: Initial seed files (32 catchers, 34 umpires) shipped via SF-7 (Wave 5).
