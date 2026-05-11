"""SF-21 caller chain: every site that calls compute_marginal_sgp or
category_gap_analysis must pass through a live config so live-standings
denominators propagate.

Wave 1 (commit 1b19b27) added an optional ``config`` parameter to both
``compute_marginal_sgp`` and ``category_gap_analysis``. Without it, the
functions silently fall back to a stale module-level ``_LC = LeagueConfig()``
singleton built from pre-season defaults. This test asserts that the five
caller sites (trade_evaluator, trade_finder, trade_intelligence x2,
trade_value, opponent_trade_analysis) thread the live config through.

Approach: ``unittest.mock.patch`` the inner call target inside the caller
site's import (because callers do ``from src.engine.portfolio.category_analysis
import category_gap_analysis``, the patched binding lives in the caller
module, not the source module). For sites where standing up the full
input plumbing is impractical, we use a structural grep test that asserts
no caller in the listed files invokes ``category_gap_analysis(...)``
without a ``config`` kwarg.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.valuation import LeagueConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


# ────────────────────────────────────────────────────────────────────
# Caller site 1: src/engine/output/trade_evaluator.py phase1
# ────────────────────────────────────────────────────────────────────


def test_trade_evaluator_threads_config_to_category_gap_analysis():
    """evaluate_trade should pass its live config (or local copy) to
    category_gap_analysis when running phase1 gap analysis."""

    cfg = LeagueConfig()
    # Mark this config object so we can verify identity downstream.
    cfg.sgp_denominators = {k: v * 7.0 for k, v in cfg.sgp_denominators.items()}

    # Build a minimal player pool — 4 hitters and 4 pitchers so all
    # downstream phases have something to chew on.
    player_pool = pd.DataFrame(
        [
            {
                "player_id": i,
                "name": f"P{i}",
                "player_name": f"P{i}",
                "team": "NYY",
                "is_hitter": 1 if i < 4 else 0,
                "positions": "OF" if i < 4 else "SP",
                "status": "active",
                "r": 70 if i < 4 else 0,
                "hr": 25 if i < 4 else 0,
                "rbi": 80 if i < 4 else 0,
                "sb": 8 if i < 4 else 0,
                "ab": 500 if i < 4 else 0,
                "h": 135 if i < 4 else 0,
                "bb": 50 if i < 4 else 0,
                "hbp": 5 if i < 4 else 0,
                "sf": 5 if i < 4 else 0,
                "pa": 560 if i < 4 else 0,
                "avg": 0.270 if i < 4 else 0,
                "obp": 0.340 if i < 4 else 0,
                "w": 0 if i < 4 else 12,
                "l": 0 if i < 4 else 7,
                "sv": 0,
                "k": 0 if i < 4 else 175,
                "era": 0 if i < 4 else 3.80,
                "whip": 0 if i < 4 else 1.20,
                "ip": 0 if i < 4 else 180,
                "er": 0 if i < 4 else 76,
                "bb_allowed": 0 if i < 4 else 50,
                "h_allowed": 0 if i < 4 else 165,
            }
            for i in range(8)
        ]
    )

    user_roster = list(range(7))  # ids 0..6, leave id 7 as FA
    giving = [0]
    receiving = [7]

    standings_df = pd.DataFrame(
        [
            {"team_name": "Team Hickey", "category": cat, "total": v, "rank": 6}
            for cat, v in {
                "R": 100,
                "HR": 30,
                "RBI": 90,
                "SB": 12,
                "AVG": 0.270,
                "OBP": 0.340,
                "W": 12,
                "L": 7,
                "SV": 5,
                "K": 175,
                "ERA": 3.80,
                "WHIP": 1.20,
            }.items()
        ]
        + [
            {"team_name": f"Team {n}", "category": cat, "total": v, "rank": 6}
            for n in ("B", "C", "D")
            for cat, v in {
                "R": 110,
                "HR": 32,
                "RBI": 95,
                "SB": 15,
                "AVG": 0.275,
                "OBP": 0.345,
                "W": 13,
                "L": 8,
                "SV": 6,
                "K": 180,
                "ERA": 3.70,
                "WHIP": 1.18,
            }.items()
        ]
    )

    # Patch the binding inside the trade_evaluator module (where it was
    # imported via `from ... import category_gap_analysis`).
    with (
        patch("src.engine.output.trade_evaluator.category_gap_analysis") as cga_mock,
        patch("src.engine.output.trade_evaluator.load_league_standings", return_value=standings_df),
    ):
        cga_mock.return_value = {
            cat: {
                "rank": 6,
                "is_punt": False,
                "marginal_value": 1.0,
                "gap_to_next": 0.0,
                "gainable_positions": 1,
            }
            for cat in cfg.all_categories
        }

        from src.engine.output.trade_evaluator import evaluate_trade

        try:
            evaluate_trade(
                giving_ids=giving,
                receiving_ids=receiving,
                user_roster_ids=user_roster,
                player_pool=player_pool,
                config=cfg,
                user_team_name="Team Hickey",
                weeks_remaining=16,
                enable_mc=False,
                enable_context=False,
                enable_game_theory=False,
                apply_ytd_blend=False,
            )
        except Exception:
            # Even if downstream plumbing blows up, we only care that the
            # mocked category_gap_analysis was invoked with config.
            pass

        assert cga_mock.called, "category_gap_analysis was never called"
        kwargs = cga_mock.call_args.kwargs
        passed_config = kwargs.get("config")
        assert passed_config is not None, (
            f"category_gap_analysis was called without a config kwarg. call_args={cga_mock.call_args}"
        )
        # Must be the live config we built (or a derivative of it),
        # not the module-level singleton with default denominators.
        assert passed_config.sgp_denominators["R"] == cfg.sgp_denominators["R"], (
            f"Passed config has stale denominators: {passed_config.sgp_denominators['R']} "
            f"vs expected {cfg.sgp_denominators['R']}"
        )


# ────────────────────────────────────────────────────────────────────
# Caller site 2: src/trade_finder.py _compute_user_category_profile
# ────────────────────────────────────────────────────────────────────


def test_trade_finder_user_category_profile_passes_config():
    """_compute_user_category_profile must thread its config to
    category_gap_analysis."""
    cfg = LeagueConfig()
    cfg.sgp_denominators = {k: v * 5.0 for k, v in cfg.sgp_denominators.items()}

    all_team_totals = {
        "Team Hickey": {
            "R": 100,
            "HR": 30,
            "RBI": 90,
            "SB": 12,
            "AVG": 0.270,
            "OBP": 0.340,
            "W": 12,
            "L": 7,
            "SV": 5,
            "K": 175,
            "ERA": 3.80,
            "WHIP": 1.20,
        },
        "Team B": {
            "R": 110,
            "HR": 32,
            "RBI": 95,
            "SB": 15,
            "AVG": 0.275,
            "OBP": 0.345,
            "W": 13,
            "L": 8,
            "SV": 6,
            "K": 180,
            "ERA": 3.70,
            "WHIP": 1.18,
        },
    }

    # _compute_user_category_profile imports category_gap_analysis
    # locally inside the function. Patch where the source defines it
    # (since the binding is created at call time, not module load).
    with patch("src.engine.portfolio.category_analysis.category_gap_analysis") as cga_mock:
        cga_mock.return_value = {}
        from src.trade_finder import _compute_user_category_profile

        _compute_user_category_profile("Team Hickey", all_team_totals, cfg, weeks_remaining=10)

        assert cga_mock.called, "category_gap_analysis was never called"
        kwargs = cga_mock.call_args.kwargs
        passed_config = kwargs.get("config")
        assert passed_config is not None, (
            f"category_gap_analysis was called without a config kwarg. call_args={cga_mock.call_args}"
        )
        assert passed_config.sgp_denominators["R"] == cfg.sgp_denominators["R"]


# ────────────────────────────────────────────────────────────────────
# Caller site 3: src/trade_intelligence.py — both call sites
# ────────────────────────────────────────────────────────────────────


def test_trade_intelligence_get_category_weights_passes_config():
    """get_category_weights must thread config through to category_gap_analysis."""
    cfg = LeagueConfig()
    cfg.sgp_denominators = {k: v * 3.0 for k, v in cfg.sgp_denominators.items()}

    all_team_totals = {
        "Team Hickey": {"R": 100, "HR": 30, "RBI": 90, "SB": 12},
        "Team B": {"R": 110, "HR": 32, "RBI": 95, "SB": 15},
    }

    with patch("src.engine.portfolio.category_analysis.category_gap_analysis") as cga_mock:
        cga_mock.return_value = {}
        from src.trade_intelligence import get_category_weights

        get_category_weights("Team Hickey", all_team_totals, cfg, weeks_remaining=10)

        assert cga_mock.called
        passed_config = cga_mock.call_args.kwargs.get("config")
        assert passed_config is not None, (
            f"category_gap_analysis called without config kwarg. call_args={cga_mock.call_args}"
        )
        assert passed_config.sgp_denominators["R"] == cfg.sgp_denominators["R"]


def test_trade_intelligence_get_user_gap_analysis_passes_config():
    """_get_user_gap_analysis must accept and thread config to
    category_gap_analysis."""
    cfg = LeagueConfig()
    cfg.sgp_denominators = {k: v * 4.0 for k, v in cfg.sgp_denominators.items()}

    all_team_totals = {
        "Team Hickey": {"R": 100, "HR": 30},
        "Team B": {"R": 110, "HR": 32},
    }

    with patch("src.engine.portfolio.category_analysis.category_gap_analysis") as cga_mock:
        cga_mock.return_value = {}
        from src.trade_intelligence import _get_user_gap_analysis

        _get_user_gap_analysis("Team Hickey", all_team_totals, weeks_remaining=10, config=cfg)

        assert cga_mock.called
        passed_config = cga_mock.call_args.kwargs.get("config")
        assert passed_config is not None
        assert passed_config.sgp_denominators["R"] == cfg.sgp_denominators["R"]


# ────────────────────────────────────────────────────────────────────
# Caller site 4: src/trade_value.py compute_contextual_values
# ────────────────────────────────────────────────────────────────────


def test_trade_value_contextual_passes_config():
    """compute_contextual_values must thread config to category_gap_analysis."""
    cfg = LeagueConfig()
    cfg.sgp_denominators = {k: v * 6.0 for k, v in cfg.sgp_denominators.items()}

    trade_values = pd.DataFrame(
        [
            {
                "player_id": 1,
                "trade_value": 50.0,
                "tier": "B",
            }
        ]
    )

    user_totals = {"R": 100, "HR": 30, "RBI": 90, "SB": 12}
    all_team_totals = {
        "Team Hickey": user_totals,
        "Team B": {"R": 110, "HR": 32, "RBI": 95, "SB": 15},
    }

    with patch("src.engine.portfolio.category_analysis.category_gap_analysis") as cga_mock:
        cga_mock.return_value = {}
        from src.trade_value import compute_contextual_values

        compute_contextual_values(
            trade_values=trade_values,
            user_totals=user_totals,
            all_team_totals=all_team_totals,
            user_team_name="Team Hickey",
            config=cfg,
        )

        assert cga_mock.called
        passed_config = cga_mock.call_args.kwargs.get("config")
        assert passed_config is not None
        assert passed_config.sgp_denominators["R"] == cfg.sgp_denominators["R"]


# ────────────────────────────────────────────────────────────────────
# Caller site 5: src/opponent_trade_analysis.py compute_opponent_needs
# ────────────────────────────────────────────────────────────────────


def test_opponent_trade_analysis_passes_config():
    """compute_opponent_needs must accept and thread config to
    category_gap_analysis. Wave 1's stale comment ('config is not accepted
    here') must also be removed since the API now accepts config."""
    cfg = LeagueConfig()
    cfg.sgp_denominators = {k: v * 8.0 for k, v in cfg.sgp_denominators.items()}

    all_team_totals = {
        "Opp Team": {"R": 100, "HR": 30, "RBI": 90, "SB": 12},
        "Other": {"R": 110, "HR": 32, "RBI": 95, "SB": 15},
    }

    with patch("src.engine.portfolio.category_analysis.category_gap_analysis") as cga_mock:
        cga_mock.return_value = {}
        from src.opponent_trade_analysis import compute_opponent_needs

        compute_opponent_needs("Opp Team", all_team_totals, weeks_remaining=10, config=cfg)

        assert cga_mock.called
        passed_config = cga_mock.call_args.kwargs.get("config")
        assert passed_config is not None
        assert passed_config.sgp_denominators["R"] == cfg.sgp_denominators["R"]


def test_opponent_trade_analysis_stale_comment_removed():
    """The misleading comment claiming category_gap_analysis() does not
    accept config must be removed/updated, since Wave 1 added the parameter."""
    src = (REPO_ROOT / "src" / "opponent_trade_analysis.py").read_text()
    assert "config is not accepted here because category_gap_analysis()" not in src, (
        "Stale comment still present in src/opponent_trade_analysis.py — "
        "category_gap_analysis() now accepts config (Wave 1 / commit 1b19b27)."
    )


# ────────────────────────────────────────────────────────────────────
# Structural smoke test: no caller invokes category_gap_analysis(...)
# without a config= kwarg in the listed source files.
# ────────────────────────────────────────────────────────────────────


CALLER_FILES = [
    "src/engine/output/trade_evaluator.py",
    "src/trade_finder.py",
    "src/trade_intelligence.py",
    "src/trade_value.py",
    "src/opponent_trade_analysis.py",
]


def _extract_call(text: str, fn_name: str, start: int) -> str | None:
    """Walk the text from the position of ``fn_name`` and return the full
    call expression up to the matching ``)``. Returns None if the start
    position is not the function call site (e.g. it's a definition or
    string literal)."""
    # Find the '(' right after the name
    paren_start = text.find("(", start + len(fn_name))
    if paren_start == -1:
        return None
    depth = 1
    i = paren_start + 1
    in_str: str | None = None
    while i < len(text) and depth > 0:
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == in_str:
                in_str = None
        else:
            if ch in ('"', "'"):
                in_str = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start:i]


def _strip_strings_and_comments(text: str) -> str:
    """Strip Python string literals and comments so a regex sees only code.

    Preserves source positions by replacing string/comment characters with
    spaces, so line/column references stay usable elsewhere if needed.
    """
    out = list(text)
    i = 0
    n = len(text)
    in_str: str | None = None
    triple = False
    while i < n:
        ch = text[i]
        nxt2 = text[i : i + 3]
        if in_str:
            if triple:
                if nxt2 == in_str * 3:
                    out[i : i + 3] = [" "] * 3
                    in_str = None
                    triple = False
                    i += 3
                    continue
            else:
                if ch == "\\" and i + 1 < n:
                    out[i] = " "
                    out[i + 1] = " "
                    i += 2
                    continue
                if ch == in_str:
                    out[i] = " "
                    in_str = None
                    i += 1
                    continue
            if ch != "\n":
                out[i] = " "
            i += 1
            continue
        if ch == "#":
            # Comment — strip to end of line
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if nxt2 in ('"""', "'''"):
            in_str = ch
            triple = True
            out[i : i + 3] = [" "] * 3
            i += 3
            continue
        if ch in ('"', "'"):
            in_str = ch
            triple = False
            out[i] = " "
            i += 1
            continue
        i += 1
    return "".join(out)


def test_no_caller_invokes_category_gap_analysis_without_config():
    """Smoke test: every invocation of ``category_gap_analysis(...)`` in the
    listed caller files must include ``config=`` in the kwargs."""
    bad_calls: list[tuple[str, str]] = []
    fn = "category_gap_analysis"
    # Match call sites only — exclude `def category_gap_analysis(`
    # and bare `import category_gap_analysis`.
    pattern = re.compile(r"\bcategory_gap_analysis\b")

    for rel in CALLER_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        raw = path.read_text()
        # Strip strings and comments so docstrings / log messages don't
        # get flagged as missing-config call sites.
        text = _strip_strings_and_comments(raw)
        for m in pattern.finditer(text):
            start = m.start()
            # Skip if this is the import or the def itself
            line_start = text.rfind("\n", 0, start) + 1
            line_prefix = text[line_start:start].strip()
            if line_prefix.startswith("def ") or line_prefix.endswith("import") or "import " in line_prefix:
                continue
            # Skip if part of a multi-line import: walk backward through
            # whitespace and commas — if we hit "import" before a non-
            # import token, this is an import.
            preceding = text[:start].rstrip()
            if preceding.endswith(",") or preceding.endswith("("):
                # Possibly inside a parenthesized import — peek further back
                # for "import" keyword on the line that opened the paren.
                open_paren = preceding.rfind("(")
                if open_paren != -1:
                    paren_line_start = text.rfind("\n", 0, open_paren) + 1
                    paren_line = text[paren_line_start:open_paren]
                    if "import" in paren_line:
                        continue
            # Skip if next non-space char after name is not '('
            next_idx = start + len(fn)
            while next_idx < len(text) and text[next_idx] in (" ", "\t", "\n"):
                next_idx += 1
            if next_idx >= len(text) or text[next_idx] != "(":
                continue
            call = _extract_call(text, fn, start)
            if call is None:
                continue
            if "config=" not in call:
                bad_calls.append((rel, call.replace("\n", " ")[:200]))

    assert not bad_calls, (
        "Found category_gap_analysis(...) calls without a config kwarg "
        "in scoped files:\n  " + "\n  ".join(f"{f}: {c}" for f, c in bad_calls)
    )
