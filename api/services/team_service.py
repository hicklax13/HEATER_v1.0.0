"""The ONE module in the API package allowed to call the engines.

Maps existing engine output -> the My Team contract. Engine calls mirror the
canonical signatures in CLAUDE.md (get_yahoo_data_service, load_player_pool,
resolve_viewer_team_name, MatchupContextService). Kept resilient: any missing
live data degrades to an empty/None field rather than raising."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from api.contracts.common import StatItem
from api.contracts.my_team import (
    CategoryLine,
    Lever,
    LeverPickup,
    MatchupHero,
    Mover,
    MyTeamResponse,
    StatusChip,
)
from api.services.player_ref import player_ref_from_pool

logger = logging.getLogger(__name__)

# IL/DTD statuses that count toward roster-health (mirrors src.alerts). Yahoo also
# emits free-form variants like "IL10 - 3 days", so prefixes are matched too.
_IL_STATUSES = {"IL", "IL10", "IL15", "IL60", "DTD", "NA", "OUT"}
_IL_PREFIXES = ("IL10", "IL15", "IL60", "IL ", "DTD")
# Freshness is reported from the most-relevant live sources for this page.
_FRESHNESS_SOURCES = ("yahoo_standings", "season_stats", "yahoo_rosters")
_PLAYOFF_CUT_FALLBACK = 4


def _is_il_status(status: str) -> bool:
    """True for an IL/DTD roster status, incl. Yahoo free-form ('IL10 - 3 days')."""
    s = str(status).upper().strip()
    return s in _IL_STATUSES or any(s.startswith(p) for p in _IL_PREFIXES)


def _f(value, default: float = 0.0) -> float:
    """Finite-float coercion (None/NaN/inf/junk → default)."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(fval) or math.isinf(fval)) else fval


def _avg_value(value) -> str:
    """Batting-average display VALUE: strip the leading zero for a true .000-.999 rate
    ('.310'), but keep it for 0/NaN ('0.000') so it never looks like a parse artifact."""
    fval = _f(value)
    text = f"{fval:.3f}"
    return text.lstrip("0") if 0.0 < fval < 1.0 else text


def _stat_value(value, cat: str) -> str:
    """Format a YTD value for display: rate cats keep decimals, counting cats round to int."""
    cu = cat.upper()
    if cu in ("AVG", "OBP"):
        return _avg_value(value)
    if cu in ("ERA", "WHIP"):
        return f"{_f(value):.2f}"
    return str(int(round(_f(value))))


def _mover_stats(row, hitter: bool) -> list[StatItem]:
    """Two YTD stats for a mover as StatItem{label,value}, by player type (pool ytd_* cols)."""
    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    if hitter:
        return [
            StatItem(label="HR", value=_stat_value(g("ytd_hr"), "HR")),
            StatItem(label="AVG", value=_stat_value(g("ytd_avg"), "AVG")),
        ]
    return [
        StatItem(label="K", value=_stat_value(g("ytd_k"), "K")),
        StatItem(label="ERA", value=_stat_value(g("ytd_era"), "ERA")),
    ]


def _cat_stat(pool_row, cat: str) -> StatItem:
    """A lever pickup's stat in the lever category, e.g. {label:"SB", value:"24"}.
    Reads ytd_<cat> from the player's pool row; NaN/missing → '0'."""
    g = (
        (pool_row.get if hasattr(pool_row, "get") else (lambda k, d=None: None))
        if pool_row is not None
        else (lambda k, d=None: None)
    )
    return StatItem(label=cat, value=_stat_value(g(f"ytd_{cat.lower()}"), cat))


class TeamService:
    def get_my_team(self, team_name: str) -> MyTeamResponse:
        from src.valuation import LeagueConfig
        from src.yahoo_data_service import get_yahoo_data_service

        yds = get_yahoo_data_service()
        cfg = LeagueConfig()
        raw_matchup = yds.get_matchup()
        standings = yds.get_standings()
        rank, record = self._rank_and_record(standings, team_name)
        week = int(raw_matchup.get("week", 0)) if raw_matchup else 0
        n_teams = self._team_count(standings)

        roster = self._roster(team_name)
        roster_ids = self._roster_ids(roster)

        return MyTeamResponse(
            team_name=team_name,
            record=record,
            rank=rank,
            matchup=self._matchup(raw_matchup, cfg),
            categories=self._categories(raw_matchup, cfg),
            eyebrow=self._eyebrow(team_name, week),
            subline=self._subline(record, rank, n_teams, standings),
            freshness_minutes=self._freshness_minutes(),
            playoff_cut_rank=self._playoff_cut_rank(),
            status_chips=self._status_chips(roster, roster_ids),
            movers=self._movers(roster_ids, cfg),
            movers_scope="mine",
            lever=self._lever(team_name, cfg),
        )

    def _lever(self, team_name: str, cfg) -> Lever | None:
        """The biggest category weakness + up to 3 FA pickups that address it.

        Mirrors fa_pool_service: build_optimizer_context → weakest cat = most-negative
        category_gap → rank_free_agents filtered to best_category == that cat. Returns
        None on cold env / no gaps (never raises). NOTE: build_optimizer_context is a
        heavy call (~2-4s, same as the Players page) — acceptable for a dashboard."""
        try:
            from src.in_season import rank_free_agents
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.yahoo_data_service import get_yahoo_data_service

            ctx = build_optimizer_context(
                scope="rest_of_season",
                yds=get_yahoo_data_service(),
                config=cfg,
                user_team_name=team_name,
                level_filter="MLB only",
            )
            gaps = ctx.category_gaps or {}
            if not gaps:
                return None
            raw_cat = min(gaps, key=gaps.get)  # most-negative gap = weakest category
            if _f(gaps.get(raw_cat), 0.0) >= 0:
                return None  # at-or-ahead in every category — no weakness to flag (lever=None)
            # category_gaps keys are lowercase ("era") but rank_free_agents.best_category
            # is UPPERCASE ("ERA") — normalize so the pickup filter actually matches.
            cat = str(raw_cat).upper()
            behind_by = round(abs(_f(gaps.get(raw_cat))), 1)
            return Lever(
                category_key=cat,
                headline=f"{cat} is your weakest category",
                behind_by=behind_by,
                pickups=self._lever_pickups(ctx, cat, cfg, rank_free_agents),
            )
        except Exception as exc:
            logger.warning("TeamService._lever failed: %s", exc)
            return None

    @staticmethod
    def _lever_pickups(ctx, cat: str, cfg, rank_free_agents) -> list[LeverPickup]:
        """Top-3 FAs whose best_category is the lever category, enriched + with proj_stat."""
        try:
            if ctx.free_agents is None or ctx.free_agents.empty or ctx.player_pool.empty:
                return []
            ranked = rank_free_agents(ctx.user_roster_ids, ctx.free_agents, ctx.player_pool, cfg)
            if ranked is None or ranked.empty or "best_category" not in ranked.columns:
                return []
            matches = ranked[ranked["best_category"] == cat].head(3)
            pool = ctx.player_pool
            out: list[LeverPickup] = []
            for r in matches.to_dict("records"):
                pid = int(r.get("player_id", 0) or 0)
                prow = None
                try:
                    m = pool[pool["player_id"] == pid]
                    if not m.empty:
                        prow = m.iloc[0]
                except Exception:
                    prow = None
                out.append(
                    LeverPickup(
                        player=player_ref_from_pool(pid, pool, name=r.get("player_name"), positions=r.get("positions")),
                        proj_stat=_cat_stat(prow, cat),
                    )
                )
            return out
        except Exception as exc:
            logger.warning("TeamService._lever_pickups failed: %s", exc)
            return []

    # ── existing core ────────────────────────────────────────────────────
    @staticmethod
    def _rank_and_record(standings, team_name: str) -> tuple[int, str]:
        # standings carries per-category rows; the WINS category holds the W-L.
        try:
            wins = standings[(standings["team_name"] == team_name) & (standings["category"] == "WINS")]
            rank = int(wins["rank"].iloc[0]) if not wins.empty else 0
            record = str(wins["total"].iloc[0]) if not wins.empty else "0-0-0"
        except Exception:
            rank, record = 0, "0-0-0"
        return rank, record

    @staticmethod
    def _matchup(raw_matchup: dict | None, cfg) -> MatchupHero | None:
        if not raw_matchup:
            return None
        return MatchupHero(
            opponent=str(raw_matchup.get("opponent", "")),
            week=int(raw_matchup.get("week", 0)),
            win_prob=float(raw_matchup.get("win_prob", 0.0)),
            tie_prob=float(raw_matchup.get("tie_prob", 0.0)),
            loss_prob=float(raw_matchup.get("loss_prob", 0.0)),
        )

    @staticmethod
    def _categories(raw_matchup: dict | None, cfg) -> list[CategoryLine]:
        if not raw_matchup or "categories" not in raw_matchup:
            return []
        inverse = set(cfg.inverse_stats)
        out: list[CategoryLine] = []
        for c in raw_matchup["categories"]:
            cat = str(c.get("cat", ""))
            you = float(c.get("you", 0.0))
            opp = float(c.get("opp", 0.0))
            out.append(
                CategoryLine(
                    cat=cat,
                    you=you,
                    opp=opp,
                    edge=you - opp,
                    win_prob=float(c.get("win_prob", 0.0)),
                    inverse=cat in inverse,
                )
            )
        return out

    # ── slice 1 additions ────────────────────────────────────────────────
    @staticmethod
    def _roster(team_name: str):
        """The user's league-roster rows (player_id/status/name); empty on any failure."""
        import pandas as pd

        try:
            from src.database import load_league_rosters

            lr = load_league_rosters()
            if lr is None or lr.empty or "team_name" not in lr.columns:
                return pd.DataFrame()
            return lr[lr["team_name"] == team_name].copy()
        except Exception as exc:
            logger.warning("TeamService._roster failed: %s", exc)
            return pd.DataFrame()

    @staticmethod
    def _roster_ids(roster) -> list[int]:
        try:
            if roster is None or roster.empty or "player_id" not in roster.columns:
                return []
            ids = []
            for v in roster["player_id"].tolist():
                try:
                    ids.append(int(v))
                except (TypeError, ValueError):
                    continue
            return ids
        except Exception:
            return []

    @staticmethod
    def _team_count(standings) -> int:
        try:
            return int(standings["team_name"].nunique()) if standings is not None and not standings.empty else 12
        except Exception:
            return 12

    def _movers(self, roster_ids: list[int], cfg) -> list[Mover]:
        """Hot/cold players on the user's roster (trend vs projection), top 4 by |delta|.

        Filters the pool to the roster FIRST — compute_player_trends computes each
        player's own actual-vs-projected delta (not pool-relative), so slicing to the
        ~28 roster players is both correct and fast (no full-pool scan)."""
        if not roster_ids:
            return []
        try:
            from src.database import load_player_pool, load_season_stats
            from src.trend_tracker import compute_player_trends

            pool = load_player_pool()
            if pool is None or pool.empty:
                return []
            pool = pool.rename(columns={"name": "player_name"}) if "name" in pool.columns else pool
            roster_pool = pool[pool["player_id"].isin(roster_ids)]
            if roster_pool.empty:
                return []
            season = load_season_stats()
            if season is not None and not season.empty and "player_id" in season.columns:
                season = season[season["player_id"].isin(roster_ids)]
            trended = compute_player_trends(roster_pool, season, cfg)
            if trended is None or trended.empty or "trend_label" not in trended.columns:
                return []
            movers_df = trended[trended["trend_label"].isin(["HOT", "COLD"])].copy()
            if movers_df.empty:
                return []
            movers_df["_abs"] = movers_df["trend_delta"].map(lambda v: abs(_f(v)))
            movers_df = movers_df.sort_values("_abs", ascending=False).head(4)
            return [self._to_mover(r, roster_pool) for _, r in movers_df.iterrows()]
        except Exception as exc:
            logger.warning("TeamService._movers failed: %s", exc)
            return []

    @staticmethod
    def _to_mover(row, pool) -> Mover:
        g = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
        try:
            pid = int(g("player_id", 0) or 0)
        except (TypeError, ValueError):
            pid = 0
        ih = g("is_hitter", True)
        hitter = True if (isinstance(ih, float) and math.isnan(ih)) else bool(ih)
        hot = str(g("trend_label", "")).upper() == "HOT"
        return Mover(
            player=player_ref_from_pool(pid, pool, name=g("player_name") or g("name"), positions=g("positions")),
            stats=_mover_stats(row, hitter),
            trend="up" if hot else "down",
            tag="hot" if hot else "cold",
            context="Trending hot vs projection" if hot else "Cooling off vs projection",
            rostered_by_you=True,
        )

    def _status_chips(self, roster, roster_ids: list[int]) -> list[StatusChip]:
        chips: list[StatusChip] = []
        il = self._il_count(roster)
        if il is not None:
            chips.append(StatusChip(label="IL", value=il, status="warn" if il else "ok"))
        news = self._news_count(roster_ids)
        if news is not None:
            chips.append(StatusChip(label="News", value=news, status="info"))
        return chips

    @staticmethod
    def _il_count(roster) -> int | None:
        try:
            if roster is None or roster.empty or "status" not in roster.columns:
                return None
            return int(roster["status"].fillna("").map(_is_il_status).sum())
        except Exception:
            return None

    @staticmethod
    def _news_count(roster_ids: list[int]) -> int | None:
        if not roster_ids:
            return None
        try:
            from src.database import get_connection

            conn = get_connection()
            try:
                placeholders = ",".join("?" * len(roster_ids))
                cur = conn.execute(
                    f"SELECT COUNT(*) FROM player_news WHERE player_id IN ({placeholders})",
                    roster_ids,
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()
        except Exception:
            return None

    @staticmethod
    def _freshness_minutes() -> float | None:
        """Minutes since the STALEST core live source last refreshed (None if unknown).

        The dashboard is only as fresh as its oldest core input, so report the max
        age — a small standings age must not mask hours-stale season stats. Clamped at
        0 so clock skew / a future timestamp can't yield a negative age."""
        try:
            from src.database import get_refresh_log_snapshot

            snap = get_refresh_log_snapshot()
            if not snap:
                return None
            now = datetime.now(UTC)
            ages: list[float] = []
            for rec in snap:
                if rec.get("source") not in _FRESHNESS_SOURCES:
                    continue
                ts_raw = rec.get("last_refresh")
                if not ts_raw:
                    continue
                try:
                    ts = datetime.fromisoformat(str(ts_raw))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    ages.append(max(0.0, (now - ts).total_seconds() / 60.0))
                except (TypeError, ValueError):
                    continue
            return round(max(ages), 1) if ages else None
        except Exception:
            return None

    @staticmethod
    def _playoff_cut_rank() -> int:
        try:
            from src.engine.output.playoff_sim import _PLAYOFF_SPOTS

            return int(_PLAYOFF_SPOTS)
        except Exception:
            return _PLAYOFF_CUT_FALLBACK

    @staticmethod
    def _eyebrow(team_name: str, week: int) -> str:
        parts = ["Season"]
        if week:
            parts.append(f"Week {week}")
        if team_name:
            parts.append(team_name)
        return " · ".join(parts)

    def _subline(self, record: str, rank: int, n_teams: int, standings) -> str:
        parts: list[str] = []
        if record:
            parts.append(record)
        if rank and n_teams:
            parts.append(f"{self._ordinal(rank)} of {n_teams}")
        gb = self._games_back(standings, record)
        if gb is not None and gb > 0:
            parts.append(f"{gb:g} GB from 1st")
        return " · ".join(parts)

    @staticmethod
    def _ordinal(n: int) -> str:
        if 10 <= (n % 100) <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    @staticmethod
    def _wins_from_record(record: str) -> float | None:
        try:
            return float(str(record).split("-")[0])
        except (TypeError, ValueError, IndexError):
            return None

    def _games_back(self, standings, record: str) -> float | None:
        """Matchup wins behind the league leader (H2H, simple win delta). None if not derivable."""
        try:
            your_wins = self._wins_from_record(record)
            if your_wins is None or standings is None or standings.empty:
                return None
            wins_rows = standings[standings["category"] == "WINS"]
            if wins_rows.empty:
                return None
            leader_wins = max(
                (w for w in (self._wins_from_record(t) for t in wins_rows["total"]) if w is not None),
                default=None,
            )
            if leader_wins is None:
                return None
            return round(leader_wins - your_wins, 1)
        except Exception:
            return None
