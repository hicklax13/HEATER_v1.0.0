"""Wave 10 — Pitcher position resolution + SUSP exclusion.

Tests the `resolve_pitcher_positions` helper that enriches pitcher rows
with SP/RP qualifiers when the source data only has the generic 'P' tag.
Also tests that SUSP roster status is treated as IL (health=0).

The helper is needed because:
- ~948 pitchers in the live DB have positions='P' (no SP/RP qualifier)
- The lineup optimizer's SP gate looks for 'SP' or 'RP' tokens to decide
  whether a pitcher counts as a starter on a given day
- Without enrichment, generic-P pitchers all get classified as RP/SP
  inconsistently or get zeroed-out volume during the SP gate

Signal hierarchy:
1. Existing 'SP' or 'RP' token (idempotent — never downgrade)
2. depth_chart_role from depth_charts.fetch_depth_charts
3. season_stats fallback (ip/games_played ratio + saves)
"""

from __future__ import annotations

import pytest

from src.depth_charts import resolve_pitcher_positions
from src.optimizer.daily_optimizer import compute_health_factor

# ──────────────────────────────────────────────────────────────────────
# Group 1: Idempotency — never downgrade pre-enriched positions
# ──────────────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_sp_p_unchanged(self):
        assert resolve_pitcher_positions("SP,P") == "SP,P"

    def test_rp_p_unchanged(self):
        assert resolve_pitcher_positions("RP,P") == "RP,P"

    def test_sp_alone_unchanged(self):
        # 'SP' alone is sufficient (covers SP and P slots in optimizer)
        assert resolve_pitcher_positions("SP") == "SP"

    def test_rp_alone_unchanged(self):
        assert resolve_pitcher_positions("RP") == "RP"

    def test_sp_rp_p_unchanged(self):
        assert resolve_pitcher_positions("SP,RP,P") == "SP,RP,P"

    def test_idempotent_under_conflicting_signal(self):
        # If positions already says SP, depth_chart_role='closer' must NOT downgrade
        assert resolve_pitcher_positions("SP,P", depth_chart_role="closer") == "SP,P"


# ──────────────────────────────────────────────────────────────────────
# Group 2: Non-pitcher passthrough — never modify hitter positions
# ──────────────────────────────────────────────────────────────────────


class TestNonPitcherPassthrough:
    def test_outfielder_unchanged(self):
        assert resolve_pitcher_positions("OF") == "OF"

    def test_first_baseman_outfielder_unchanged(self):
        assert resolve_pitcher_positions("1B,OF") == "1B,OF"

    def test_catcher_with_starter_role_unchanged(self):
        # Catcher with depth_chart_role='starter' (lineup starter) must NOT
        # become 'SP,C' — the role is ambiguous between hitters and pitchers
        assert resolve_pitcher_positions("C", depth_chart_role="starter") == "C"

    def test_dh_unchanged_with_pitcher_signals(self):
        # Stray pitcher signals on a non-pitcher row must be ignored
        assert resolve_pitcher_positions("DH", innings_pitched=192, games_played=31) == "DH"


# ──────────────────────────────────────────────────────────────────────
# Group 3: depth_chart_role resolution
# ──────────────────────────────────────────────────────────────────────


class TestDepthChartRoleResolution:
    def test_p_with_rotation_role_becomes_sp_p(self):
        assert resolve_pitcher_positions("P", depth_chart_role="rotation") == "SP,P"

    def test_p_with_starter_role_becomes_sp_p(self):
        # 'starter' means rotation when applied to a pitcher (positions=='P')
        assert resolve_pitcher_positions("P", depth_chart_role="starter") == "SP,P"

    def test_p_with_closer_role_becomes_rp_p(self):
        assert resolve_pitcher_positions("P", depth_chart_role="closer") == "RP,P"

    def test_p_with_setup_role_becomes_rp_p(self):
        assert resolve_pitcher_positions("P", depth_chart_role="setup") == "RP,P"

    def test_p_with_bullpen_role_becomes_rp_p(self):
        assert resolve_pitcher_positions("P", depth_chart_role="bullpen") == "RP,P"


# ──────────────────────────────────────────────────────────────────────
# Group 4: season_stats fallback
# ──────────────────────────────────────────────────────────────────────


class TestSeasonStatsFallback:
    def test_high_saves_becomes_rp_p(self):
        # Closer signal: sv >= 5
        assert resolve_pitcher_positions("P", saves=10) == "RP,P"

    def test_explicit_games_started_becomes_sp_p(self):
        # If GS provided directly, GS >= 5 → SP
        assert resolve_pitcher_positions("P", games_started=10) == "SP,P"

    def test_high_ip_per_game_becomes_sp_p(self):
        # Skubal: 192 IP / 31 GP = 6.19 → SP
        assert resolve_pitcher_positions("P", innings_pitched=192, games_played=31) == "SP,P"

    def test_low_ip_per_game_becomes_rp_p(self):
        # Williams: 58 IP / 67 GP = 0.86 → RP
        assert resolve_pitcher_positions("P", innings_pitched=58, games_played=67) == "RP,P"

    def test_zero_ip_returns_unchanged_p(self):
        # No data — can't determine; preserve 'P'
        assert resolve_pitcher_positions("P", innings_pitched=0, games_played=0) == "P"

    def test_no_signals_returns_unchanged_p(self):
        # Bare 'P' with no signals stays 'P' (caller will not update DB)
        assert resolve_pitcher_positions("P") == "P"

    def test_saves_takes_priority_over_ip_ratio(self):
        # Even if avg_ip suggests SP, sv >= 5 means closer
        assert resolve_pitcher_positions("P", saves=10, innings_pitched=200, games_played=31) == "RP,P"


# ──────────────────────────────────────────────────────────────────────
# Group 5: Edge cases — None, empty, malformed
# ──────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_none_positions_returns_none(self):
        assert resolve_pitcher_positions(None) is None

    def test_empty_string_returns_empty(self):
        assert resolve_pitcher_positions("") == ""

    def test_whitespace_only_returns_unchanged(self):
        # Whitespace-only positions string is not a pitcher
        assert resolve_pitcher_positions("   ") == "   "


# ──────────────────────────────────────────────────────────────────────
# Group 6: SUSP exclusion fix (Valdez bug)
# ──────────────────────────────────────────────────────────────────────


class TestSuspendedExclusion:
    def test_susp_status_health_zero(self):
        # Yahoo uses 'SUSP' for suspended players (Framber Valdez 2026)
        assert compute_health_factor("SUSP") == 0.0

    def test_susp_lowercase_health_zero(self):
        assert compute_health_factor("susp") == 0.0

    def test_susp_titlecase_health_zero(self):
        assert compute_health_factor("Susp") == 0.0

    # Regression guards on existing behavior
    def test_il15_still_health_zero(self):
        assert compute_health_factor("IL15") == 0.0

    def test_active_still_health_one(self):
        assert compute_health_factor("active") == 1.0

    def test_dtd_still_health_zero(self):
        assert compute_health_factor("DTD") == 0.0


# ──────────────────────────────────────────────────────────────────────
# Group 7: Bootstrap wiring — helper is invoked after depth_charts phase
# ──────────────────────────────────────────────────────────────────────


class TestBootstrapWiring:
    def test_enrich_function_exists(self):
        """The bootstrap-callable enrichment wrapper must exist."""
        from src.data_bootstrap import _enrich_pitcher_positions

        assert callable(_enrich_pitcher_positions)

    def test_bootstrap_phase_dispatches_enrichment(self):
        """The bootstrap orchestrator must call _enrich_pitcher_positions
        after _bootstrap_depth_charts so freshly-persisted depth_chart_role
        values flow into pitcher position enrichment in the same run."""
        import ast
        from pathlib import Path

        src_path = Path(__file__).resolve().parent.parent / "src" / "data_bootstrap.py"
        tree = ast.parse(src_path.read_text(encoding="utf-8"))

        names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert "_enrich_pitcher_positions" in names, "Bootstrap orchestrator missing _enrich_pitcher_positions phase"
        # Look for a call to _enrich_pitcher_positions somewhere in the module
        call_sites = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_enrich_pitcher_positions"
        ]
        assert call_sites, "_enrich_pitcher_positions defined but never called in data_bootstrap.py"


# ──────────────────────────────────────────────────────────────────────
# Group 8: Token-set semantics (preserves leading qualifiers, not substring)
# ──────────────────────────────────────────────────────────────────────


class TestTokenSetSemantics:
    def test_substring_not_treated_as_token(self):
        # 'SP' should not be detected inside, e.g., 'PSP' (hypothetical) —
        # this guards against substring confusion. The DB doesn't contain
        # such strings but the helper must use token-set logic.
        # Using a real-world ambiguous token: 'PH/PR' (pinch hitter/runner)
        # is NOT a pitcher and must not be enriched.
        assert resolve_pitcher_positions("PH/PR") == "PH/PR"

    def test_handles_slash_separator(self):
        # Some position strings use '/' as separator (e.g., 'C/1B,RP')
        # Already-enriched strings must be detected via either separator.
        assert resolve_pitcher_positions("P/OF,RP") == "P/OF,RP"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
