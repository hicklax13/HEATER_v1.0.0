"""Trade Finder service — the ONE place that calls find_trade_opportunities.
Maps engine output → the TradeFinderResponse contract.
Resilient: cold env / heavy-scan failure → empty suggestions, never a 500."""

from __future__ import annotations

import logging
import math

from api.contracts.common import PlayerRef
from api.contracts.trade import CategoryImpact
from api.contracts.trade_finder import TradeFinderResponse, TradeSuggestion
from api.services.player_ref import make_player_ref

logger = logging.getLogger(__name__)

# A freed roster slot in a 2-for-1 is worth a STREAMABLE replacement-level FA — NOT
# the engine's "best available FA" (src/trade_finder.py ~L701 credits the freed slot
# at the value of the single BEST FA, large enough to mask a multi-SGP value loss and
# stamp an "A" on a lopsided 2-for-1). Replacement-level (~1 SGP) is the honest credit.
_REPLACEMENT_SLOT_SGP = 1.0

# A surfaced trade must be a GENUINE gain. The need-weighted gain at/below this floor is
# noise (or a loss) → dropped. This REPLACES the old `user_sgp_gain <= 0` engine-gain
# gate, which trusted the engine's inflated gain.
_MIN_TRUE_GAIN = 0.5

# Hard safety floor on RAW (unweighted) value: never surface a trade that gives away more
# than this many SGP of total production, even if category-need weighting makes it look
# favorable. Keeps the finder from recommending "sell your strength to patch a weakness"
# deals that bleed net value. Near-even need-fit trades (raw ≈ 0) still pass.
_RAW_LOSS_FLOOR = -1.5

# How many candidates to pull from the engine, DECOUPLED from how many we display. The
# engine ranks by its inflated metric, so the genuinely-good trades rank BELOW its top
# few (live: the good fits were candidates 11-32). Pull a wide set, re-value honestly,
# then surface the best `limit`. The engine self-caps well under this for a normal roster.
_ENGINE_CANDIDATE_POOL = 60

# Small-sample guard: a player's YTD RATE stats (AVG/OBP/ERA/WHIP) are noise until they
# accumulate enough volume — a 9-IP WHIP or a 25-AB AVG can swing wildly. We scale a
# player's rate-stat SGP by min(1, volume/stabilization) so a tiny sample contributes a
# muted (not full) rate signal. Thresholds are deliberately LOW — they mute genuine
# small samples (callups, just-off-the-IL arms) without touching a 250-AB regular.
# COUNTING stats are NOT scaled: a low total already reflects the low accumulated value.
_RATE_STAB_AB = 80.0
_RATE_STAB_IP = 30.0


def _sample_reliability(row) -> float:
    """0..1 weight on a player's YTD RATE-stat signal, from their accumulated volume.
    1.0 once they clear the stabilization threshold; lower for tiny samples. Never raises."""
    try:
        if "is_hitter" in row.index:
            is_hit = bool(int(row.get("is_hitter", 1) or 0))
        else:
            is_hit = float(row.get("ytd_ab", 0) or 0) >= float(row.get("ytd_ip", 0) or 0)
        vol = float(row.get("ytd_ab" if is_hit else "ytd_ip", 0) or 0)
        stab = _RATE_STAB_AB if is_hit else _RATE_STAB_IP
        return max(0.0, min(1.0, vol / stab)) if stab > 0 else 1.0
    except Exception:
        return 1.0


def _grade_from_gain(gain: float) -> str:
    """Letter grade from the HONEST marginal true_gain (SGP). Nothing grades 'A'
    unless it's a real multi-SGP gain — the anti-"A on a value loss" ladder."""
    if gain >= 3.0:
        return "A+"
    if gain >= 2.0:
        return "A"
    if gain >= 1.2:
        return "B+"
    if gain >= 0.7:
        return "B"
    return "B-"


# Value players by their ACTUAL 2026 YTD season production — the "current total season
# stats" that the live Yahoo rankings reflect — NOT the pool's projection columns. The
# projections compress an elite hitter and a replacement-level one (Olson #23 and Moniak
# #2021 scored within 0.7 SGP); on real YTD stats the gap is honest (Olson +5.05 vs
# Moniak +2.05), so a lopsided trade reads as the loss it is. We swap the YTD columns
# into the names player_sgp reads, then fall back to the projection row ONLY when the
# player has no YTD sample yet (e.g. a yet-to-debut callup) so they aren't valued at 0.
_YTD_HIT_SWAP = (
    ("r", "ytd_r"),
    ("hr", "ytd_hr"),
    ("rbi", "ytd_rbi"),
    ("sb", "ytd_sb"),
    ("avg", "ytd_avg"),
    ("obp", "ytd_obp"),
    ("ab", "ytd_ab"),
    ("h", "ytd_h"),
    ("bb", "ytd_bb"),
    ("hbp", "ytd_hbp"),
    ("sf", "ytd_sf"),
    ("pa", "ytd_pa"),
)
_YTD_PIT_SWAP = (
    ("w", "ytd_w"),
    ("l", "ytd_l"),
    ("sv", "ytd_sv"),
    ("k", "ytd_k"),
    ("era", "ytd_era"),
    ("whip", "ytd_whip"),
    ("ip", "ytd_ip"),
    ("er", "ytd_er"),
)


def _ytd_value_row(row):
    """Return (row_for_valuation, used_ytd). Swaps the actual 2026 YTD season stats into
    the columns `SGPCalculator.player_sgp` reads so value = real production. Falls back
    to the unchanged (projection) row when the player has no YTD sample (ytd volume 0),
    so a not-yet-played player isn't valued at 0. Never raises."""
    try:
        if "is_hitter" in row.index:
            is_hit = bool(int(row.get("is_hitter", 1) or 0))
        else:
            is_hit = float(row.get("ytd_ab", 0) or 0) >= float(row.get("ytd_ip", 0) or 0)
    except Exception:
        is_hit = True
    vol_col = "ytd_ab" if is_hit else "ytd_ip"
    try:
        vol = float(row.get(vol_col, 0) or 0)
    except (TypeError, ValueError):
        vol = 0.0
    if vol <= 0:
        return row, False
    swap = _YTD_HIT_SWAP if is_hit else _YTD_PIT_SWAP
    s = row.copy()
    for col, ycol in swap:
        if ycol in row.index:
            s[col] = row.get(ycol)
    return s, True


def _category_need_weights(all_team_totals, user_team_key, config) -> dict[str, float]:
    """Per-category NEED weight for the user, from their standing among the league.

    A good H2H-categories trade improves the categories you're WEAK in — even at a small
    cost to a category you already dominate. Pure total-value (context-free) ranking
    misses that, so we weight each category's marginal delta by how much the user needs
    it: HIGHER weight where they're weak, LOWER where they're already strong. Weight =
    1 - 0.30*z (z = the user's z-score among the 12 teams in that category, inverse-stat
    aware), clamped to [0.6, 1.6] so it nudges toward fit without letting one category
    dominate. Returns all 1.0 (context-free) when the data is insufficient. Never raises."""
    import statistics

    weights: dict[str, float] = {}
    user = all_team_totals.get(user_team_key) if isinstance(all_team_totals, dict) else None
    user = user or {}
    for cat in config.all_categories:
        try:
            vals = [
                float(t[cat])
                for t in all_team_totals.values()
                if isinstance(t, dict) and isinstance(t.get(cat), (int, float))
            ]
            uval = user.get(cat)
            if len(vals) < 4 or not isinstance(uval, (int, float)):
                weights[cat] = 1.0
                continue
            sd = statistics.pstdev(vals)
            if sd < 1e-9:
                weights[cat] = 1.0
                continue
            z = (float(uval) - statistics.fmean(vals)) / sd
            if cat in config.inverse_stats:
                z = -z  # lower-is-better → flip so a high z means "strong here"
            weights[cat] = max(0.6, min(1.6, 1.0 - 0.30 * z))
        except Exception:
            weights[cat] = 1.0
    return weights


class TradeFinderService:
    def get_suggestions(self, team_name: str, limit: int = 10) -> TradeFinderResponse:
        try:
            from src.database import load_player_pool
            from src.standings_utils import get_all_team_totals
            from src.trade_finder import find_trade_opportunities
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            pool = load_player_pool()
            if pool is None or pool.empty:
                return TradeFinderResponse(team_name=team_name, reason="no_pool")

            yds = get_yahoo_data_service()
            rosters_df = yds.get_rosters()
            league_rosters = self._build_league_rosters(rosters_df)
            if not league_rosters:
                return TradeFinderResponse(team_name=team_name, reason="no_league_data")

            # Resolve the user's team against the ACTUAL roster keys (Yahoo keys
            # carry emoji/whitespace, e.g. "🏆 Team Hickey"); a raw .get(team_name)
            # missed them → empty roster → zero suggestions. Use the resolved key
            # name downstream so all_team_totals / find_trade_opportunities agree.
            resolved_team, user_roster_ids = self._resolve_user_roster(team_name, league_rosters)
            if not user_roster_ids:
                # The ORIGINAL bug. Make a recurrence LOUD, not a silent empty.
                logger.warning(
                    "TradeFinderService: team_name %r did not resolve to any roster key (have: %s)",
                    team_name,
                    list(league_rosters.keys()),
                )
                return TradeFinderResponse(team_name=team_name, reason="team_not_resolved")

            # all_team_totals is REQUIRED: find_trade_opportunities early-returns []
            # when it's None (src/trade_finder.py:1895). Compute it (Yahoo-standings-
            # first, projection fallback) and pass it.
            all_team_totals = get_all_team_totals(league_rosters=league_rosters, player_pool=pool)
            if not all_team_totals:
                # Empty totals → the engine would yield nothing. Surface it instead of
                # running a doomed scan and returning a confusing bare empty.
                logger.warning(
                    "TradeFinderService: get_all_team_totals empty for %d rosters — finder will yield no suggestions",
                    len(league_rosters),
                )
                return TradeFinderResponse(team_name=team_name, reason="no_totals")

            # Pull a WIDE candidate set (decoupled from the display limit): the engine
            # ranks by its inflated metric, so the genuinely-good trades sit below its top
            # few. Re-value all honestly, then surface the best `limit`.
            raw = find_trade_opportunities(
                user_roster_ids=user_roster_ids,
                player_pool=pool,
                config=LeagueConfig(),
                all_team_totals=all_team_totals,
                user_team_name=resolved_team,
                league_rosters=league_rosters,
                max_results=max(limit, _ENGINE_CANDIDATE_POOL),
            )

            # Category-need weights from the user's standing — so the finder favors trades
            # that improve the categories they're WEAK in (not just raw total value).
            need_weights = _category_need_weights(all_team_totals, resolved_team, LeagueConfig())
            suggestions = self._build_suggestions(raw, pool, user_roster_ids, need_weights=need_weights)[:limit]
            return TradeFinderResponse(team_name=team_name, suggestions=suggestions, reason="ok")

        except Exception as exc:
            logger.warning("TradeFinderService.get_suggestions failed: %s", exc, exc_info=True)
            return TradeFinderResponse(team_name=team_name, reason="error")

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_user_roster(team_name: str, league_rosters: dict[str, list[int]]) -> tuple[str, list[int]]:
        """Map the requested team_name to the EXACT roster key (emoji/whitespace
        tolerant), returning (resolved_key, roster_ids). Exact match wins; else a
        normalized match; else ("", []) so the caller returns empty (never a crash,
        never another team's roster)."""
        # Deferred import: api.tenancy → api.deps → this service would cycle at
        # module load. normalize_team_name is a tiny pure fn, cheap to import lazily.
        from api.tenancy import normalize_team_name

        if team_name in league_rosters:
            return team_name, league_rosters[team_name]
        target = normalize_team_name(team_name)
        for key, ids in league_rosters.items():
            if normalize_team_name(key) == target:
                return key, ids
        return "", []

    @staticmethod
    def _ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    @staticmethod
    def _partner_records() -> dict[str, str]:
        """{normalized_team_name: 'W-L-T · Nth'} from load_league_records; {} on any
        failure (records degrade to None, never crash)."""
        from api.tenancy import normalize_team_name

        try:
            from src.database import load_league_records

            df = load_league_records()
            if df is None or df.empty:
                return {}
            out: dict[str, str] = {}
            for _, r in df.iterrows():
                name = str(r.get("team_name", "") or "")
                if not name.strip():
                    continue
                w = int(r.get("wins", 0) or 0)
                loss = int(r.get("losses", 0) or 0)
                t = int(r.get("ties", 0) or 0)
                rank = int(r.get("rank", 0) or 0)
                rec = f"{w}-{loss}-{t}"
                if rank > 0:
                    rec = f"{rec} · {TradeFinderService._ordinal(rank)}"
                out[normalize_team_name(name)] = rec
            return out
        except Exception:
            logger.warning("TradeFinderService._partner_records failed", exc_info=True)
            return {}

    @staticmethod
    def _build_league_rosters(rosters_df) -> dict[str, list[int]]:
        """Convert rosters DataFrame to {team_name: [player_ids]}."""
        result: dict[str, list[int]] = {}
        if rosters_df is None:
            return result
        try:
            import pandas as pd

            if not isinstance(rosters_df, pd.DataFrame) or rosters_df.empty:
                return result
            team_col = next((c for c in ("team_name", "team") if c in rosters_df.columns), None)
            id_col = next((c for c in ("player_id", "id") if c in rosters_df.columns), None)
            if team_col is None or id_col is None:
                return result
            for team, grp in rosters_df.groupby(team_col):
                result[str(team)] = [int(v) for v in grp[id_col].dropna().tolist()]
        except Exception:
            pass
        return result

    @staticmethod
    def _player_sgp_lookup(pool):
        """Return a cached `player_id -> {cat: per-category SGP}` resolver over `pool`.
        Per-player SGP is the CANONICAL marginal-value path (`SGPCalculator.player_sgp`):
        it applies inverse-stat signs (L/ERA/WHIP) and rate-stat volume weighting
        correctly. Value is computed on ACTUAL 2026 YTD season stats (see _ytd_value_row)
        — Olson +5.05, Reynolds +4.43, Moniak +2.05 — NOT the compressed projections that
        had them within 0.7 SGP. A missing/unparseable player → {} (value 0). The id→row
        index is built ONCE; per-id results are memoized so repeated lookups are free."""
        import pandas as pd

        from src.valuation import LeagueConfig, SGPCalculator

        calc = SGPCalculator(LeagueConfig())
        rate_cats = set(calc.config.rate_stats)
        rows: dict[int, pd.Series] = {}
        if isinstance(pool, pd.DataFrame) and not pool.empty and "player_id" in pool.columns:
            # Last-write-wins on dup ids — harmless (the pool is id-unique in practice).
            for _, r in pool.iterrows():
                try:
                    rows[int(r["player_id"])] = r
                except (TypeError, ValueError):
                    continue

        cache: dict[int, dict[str, float]] = {}

        def per_cat(pid: int) -> dict[str, float]:
            try:
                key = int(pid)
            except (TypeError, ValueError):
                return {}
            if key in cache:
                return cache[key]
            row = rows.get(key)
            if row is None:
                cache[key] = {}
                return {}
            try:
                # Value by ACTUAL 2026 YTD season stats (the user's "current total season
                # stats"), falling back to projections only for a no-sample player.
                val_row, used_ytd = _ytd_value_row(row)
                raw = calc.player_sgp(val_row)
                # Guard every value: drop NaN/inf so a bad stat can't poison the sum.
                clean = {c: float(v) for c, v in raw.items() if math.isfinite(v)}
                # Small-sample guard: mute noisy YTD RATE stats (a 9-IP WHIP) by the
                # player's volume reliability. Counting stats keep their full (low) value.
                if used_ytd:
                    rel = _sample_reliability(row)
                    if rel < 1.0:
                        clean = {c: (v * rel if c in rate_cats else v) for c, v in clean.items()}
            except Exception:
                logger.warning("TradeFinderService: player_sgp failed for id=%s", key, exc_info=True)
                clean = {}
            cache[key] = clean
            return clean

        return per_cat

    @staticmethod
    def _marginal_category_impacts(giving_ids, receiving_ids, pool, per_cat=None) -> list[CategoryImpact]:
        """Per-category marginal SGP delta = Σ(receiving) − Σ(giving), per category.
        Roster-id-space INDEPENDENT (unlike the old roster-totals ratio diff, which
        depended on user_roster_ids and inflated rate stats / could flip counting
        signs — the live 'AVG +1.03'). Each player's contribution comes from the
        canonical `player_sgp` (correct inverse signs). Drops non-finite / trivial
        (<0.01) deltas. Never raises → []."""
        try:
            resolve = per_cat if per_cat is not None else TradeFinderService._player_sgp_lookup(pool)
            cat_net: dict[str, float] = {}
            for pid in receiving_ids:
                for cat, val in resolve(pid).items():
                    cat_net[cat] = cat_net.get(cat, 0.0) + val
            for pid in giving_ids:
                for cat, val in resolve(pid).items():
                    cat_net[cat] = cat_net.get(cat, 0.0) - val
            return [
                CategoryImpact(cat=k, delta=round(v, 3))
                for k, v in cat_net.items()
                if math.isfinite(v) and abs(v) >= 0.01
            ]
        except Exception:
            logger.warning("TradeFinderService._marginal_category_impacts failed", exc_info=True)
            return []

    @staticmethod
    def _build_suggestions(
        raw: list[dict],
        pool,
        user_roster_ids: list[int] | None = None,
        need_weights: dict[str, float] | None = None,
    ) -> list[TradeSuggestion]:
        """Map engine output dicts → TradeSuggestion list, RE-VALUED HONESTLY.

        The engine (src/trade_finder.py — SHARED with Streamlit, not modified here)
        values trades by a weighted roster-totals diff PLUS a freed-roster-spot bonus
        credited at the BEST available FA, then grades the result. That bonus is large
        enough to stamp an "A / You win" on a lopsided 2-for-1 that LOSES several SGP
        (the live "give Reynolds+Olson, get Moniak" card). So the service does NOT
        trust the engine's `user_sgp_gain` / `grade`. It re-values every suggestion by
        per-player marginal SGP:

          give_total = Σ sum(player_sgp(pid))  over giving_ids
          get_total  = Σ sum(player_sgp(pid))  over receiving_ids
          slot_credit = max(0, |give| − |receive|) * _REPLACEMENT_SLOT_SGP
          true_gain  = (get_total − give_total) + slot_credit

        Then:
        - FILTER: drop when true_gain <= _MIN_TRUE_GAIN (replaces the old engine-gain
          gate). A value-losing 2-for-1 is dropped, not graded.
        - net_sgp = round(true_gain, 3) — the honest value.
        - grade = _grade_from_gain(true_gain) — never the engine grade.
        - category_impacts = the marginal Σreceiving − Σgiving deltas (correct signs,
          realistic magnitudes), roster-id-space independent.
        - DEDUPE: same partner + same receiving set (give-side differs) → ONE card.

        A re-valuation failure for ONE suggestion → that suggestion is skipped, never a
        500. Honest-empty (every candidate value-losing → 0 suggestions) is acceptable.
        """
        from api.tenancy import normalize_team_name

        records = TradeFinderService._partner_records()
        per_cat = TradeFinderService._player_sgp_lookup(pool)
        suggestions: list[TradeSuggestion] = []
        seen_keys: set[tuple] = set()
        logger.info("TradeFinderDiag: re-valuing %d raw engine candidates (honest YTD)", len(raw))
        for opp in raw:
            try:
                giving_ids: list[int] = opp.get("giving_ids", [])
                receiving_ids: list[int] = opp.get("receiving_ids", [])
                partner_team: str = str(opp.get("opponent_team", ""))
                rationale: str = str(opp.get("rationale", "") or "")

                # Per-category YTD marginal deltas: Σ(receiving) − Σ(giving).
                cat_net: dict[str, float] = {}
                for pid in giving_ids:
                    for _c, _v in per_cat(pid).items():
                        cat_net[_c] = cat_net.get(_c, 0.0) - _v
                for pid in receiving_ids:
                    for _c, _v in per_cat(pid).items():
                        cat_net[_c] = cat_net.get(_c, 0.0) + _v
                slot_credit = max(0, len(giving_ids) - len(receiving_ids)) * _REPLACEMENT_SLOT_SGP
                # RAW gain = honest total value change (the safety floor guards this).
                raw_gain = sum(cat_net.values()) + slot_credit
                # NEED-WEIGHTED gain = the trade's value to the user's STANDING: each
                # category delta scaled by how much the user needs that category (1.0 each
                # when need_weights is None → context-free). This surfaces near-even trades
                # that improve weak categories while the raw floor blocks value giveaways.
                w = need_weights or {}
                weighted_gain = sum(v * w.get(c, 1.0) for c, v in cat_net.items()) + slot_credit

                logger.info(
                    "TradeFinderDiag cand give=%s recv=%s partner=%r raw=%.2f weighted=%.2f deltas=%s",
                    giving_ids,
                    receiving_ids,
                    partner_team,
                    raw_gain,
                    weighted_gain,
                    {c: round(v, 2) for c, v in cat_net.items() if abs(v) >= 0.05},
                )

                # FILTER: surface a standing-improving trade (weighted gain above the
                # floor) that doesn't bleed raw value (raw above the safety floor). The
                # live lopsided 2-for-1 fails both; a near-even need-fit passes both.
                if not math.isfinite(weighted_gain) or weighted_gain <= _MIN_TRUE_GAIN or raw_gain < _RAW_LOSS_FLOOR:
                    logger.info(
                        "TradeFinderDiag: DROP (partner=%r weighted=%.3f raw=%.3f)",
                        partner_team,
                        weighted_gain,
                        raw_gain,
                    )
                    continue

                # DEDUPE: collapse same partner + same receiving set (give-side differs).
                dedupe_key = (normalize_team_name(partner_team), tuple(sorted(receiving_ids)))
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)

                partner_record = records.get(normalize_team_name(partner_team))
                giving = _build_player_refs(giving_ids, pool)
                receiving = _build_player_refs(receiving_ids, pool)
                # Impacts come straight from the deltas we already computed (correct signs,
                # realistic magnitudes; the inflated roster-totals diff is gone).
                category_impacts = [
                    CategoryImpact(cat=k, delta=round(v, 3))
                    for k, v in cat_net.items()
                    if math.isfinite(v) and abs(v) >= 0.01
                ]

                suggestions.append(
                    TradeSuggestion(
                        partner_team=partner_team,
                        partner_record=partner_record,
                        grade=_grade_from_gain(weighted_gain),
                        giving=giving,
                        receiving=receiving,
                        net_sgp=round(weighted_gain, 3),
                        category_impacts=category_impacts,
                        rationale=rationale,
                    )
                )
            except Exception:
                # One bad suggestion must not 500 the whole list — skip it.
                logger.warning(
                    "TradeFinderService._build_suggestions: skipping a suggestion that failed re-valuation",
                    exc_info=True,
                )
                continue
        # Best trades first (by need-weighted value to the user).
        suggestions.sort(key=lambda s: s.net_sgp, reverse=True)
        logger.info("TradeFinderDiag: surfaced %d of %d candidates", len(suggestions), len(raw))
        return suggestions


def _build_player_refs(player_ids: list[int], pool) -> list[PlayerRef]:
    refs: list[PlayerRef] = []
    if pool is None:
        return refs
    try:
        import pandas as pd

        if not isinstance(pool, pd.DataFrame) or pool.empty:
            return refs
        name_col = "player_name" if "player_name" in pool.columns else "name"
        for pid in player_ids:
            row = pool[pool["player_id"] == pid]
            if row.empty:
                refs.append(PlayerRef(id=pid, name=f"Player {pid}", positions=""))
            else:
                r = row.iloc[0]
                refs.append(
                    make_player_ref(
                        id=pid,
                        name=str(r.get(name_col, f"Player {pid}") or f"Player {pid}"),
                        positions=str(r.get("positions", "") or ""),
                        mlb_id=r.get("mlb_id"),
                        team_abbr=r.get("team"),
                    )
                )
    except Exception:
        for pid in player_ids:
            refs.append(PlayerRef(id=pid, name=f"Player {pid}", positions=""))
    return refs
