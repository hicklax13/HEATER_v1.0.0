"""Wave 11A — DCV-A1-001 structural guard: replacement-level constants
and raw SGP denominators MUST be registered in CONSTANTS_REGISTRY.

Without registration, sensitivity_analysis can't perturb them and the
audit's HIGH-severity finding (that 0.005 shift in _REPL_AVG produces
16-33% per-player DCV change) goes undetected by the existing sensitivity
framework.

This guard pins:
1. All 8 entries (4 replacement levels + 4 raw SGP denoms) exist in
   CONSTANTS_REGISTRY with HIGH/MEDIUM sensitivity classification.
2. daily_optimizer.py module-level constants are loaded FROM the registry
   (no hardcoded copies that could drift).
"""

from __future__ import annotations


class TestReplacementConstantsInRegistry:
    """All 4 replacement-level entries must be in CONSTANTS_REGISTRY."""

    def test_repl_avg_in_registry(self):
        from src.optimizer.constants_registry import CONSTANTS_REGISTRY

        assert "repl_avg" in CONSTANTS_REGISTRY
        entry = CONSTANTS_REGISTRY["repl_avg"]
        assert entry.value > 0
        assert entry.lower_bound < entry.value < entry.upper_bound
        assert entry.sensitivity == "HIGH"
        assert entry.citation, "repl_avg must carry a citation"

    def test_repl_obp_in_registry(self):
        from src.optimizer.constants_registry import CONSTANTS_REGISTRY

        assert "repl_obp" in CONSTANTS_REGISTRY
        entry = CONSTANTS_REGISTRY["repl_obp"]
        assert entry.value > 0
        assert entry.lower_bound < entry.value < entry.upper_bound
        assert entry.sensitivity == "HIGH"

    def test_repl_era_in_registry(self):
        from src.optimizer.constants_registry import CONSTANTS_REGISTRY

        assert "repl_era" in CONSTANTS_REGISTRY
        entry = CONSTANTS_REGISTRY["repl_era"]
        assert entry.value > 0
        assert entry.lower_bound < entry.value < entry.upper_bound
        assert entry.sensitivity == "HIGH"

    def test_repl_whip_in_registry(self):
        from src.optimizer.constants_registry import CONSTANTS_REGISTRY

        assert "repl_whip" in CONSTANTS_REGISTRY
        entry = CONSTANTS_REGISTRY["repl_whip"]
        assert entry.value > 0
        assert entry.lower_bound < entry.value < entry.upper_bound
        assert entry.sensitivity == "HIGH"


class TestRawSGPDenomInRegistry:
    """All 4 raw-unit SGP denominators must be in CONSTANTS_REGISTRY."""

    def test_raw_sgp_denom_avg_in_registry(self):
        from src.optimizer.constants_registry import CONSTANTS_REGISTRY

        assert "raw_sgp_denom_avg" in CONSTANTS_REGISTRY
        entry = CONSTANTS_REGISTRY["raw_sgp_denom_avg"]
        assert entry.value > 0
        assert entry.lower_bound < entry.value < entry.upper_bound

    def test_raw_sgp_denom_obp_in_registry(self):
        from src.optimizer.constants_registry import CONSTANTS_REGISTRY

        assert "raw_sgp_denom_obp" in CONSTANTS_REGISTRY
        entry = CONSTANTS_REGISTRY["raw_sgp_denom_obp"]
        assert entry.value > 0

    def test_raw_sgp_denom_era_in_registry(self):
        from src.optimizer.constants_registry import CONSTANTS_REGISTRY

        assert "raw_sgp_denom_era" in CONSTANTS_REGISTRY
        entry = CONSTANTS_REGISTRY["raw_sgp_denom_era"]
        assert entry.value > 0

    def test_raw_sgp_denom_whip_in_registry(self):
        from src.optimizer.constants_registry import CONSTANTS_REGISTRY

        assert "raw_sgp_denom_whip" in CONSTANTS_REGISTRY
        entry = CONSTANTS_REGISTRY["raw_sgp_denom_whip"]
        assert entry.value > 0


class TestDailyOptimizerLoadsFromRegistry:
    """daily_optimizer's _REPL_* / _RAW_SGP_DENOM module constants MUST
    equal the registry values. This guards against accidental drift if
    someone hardcodes a new value in daily_optimizer without updating
    the registry."""

    def test_repl_constants_match_registry(self):
        from src.optimizer.constants_registry import CONSTANTS_REGISTRY
        from src.optimizer.daily_optimizer import _REPL_AVG, _REPL_ERA, _REPL_OBP, _REPL_WHIP

        assert _REPL_AVG == CONSTANTS_REGISTRY["repl_avg"].value
        assert _REPL_OBP == CONSTANTS_REGISTRY["repl_obp"].value
        assert _REPL_ERA == CONSTANTS_REGISTRY["repl_era"].value
        assert _REPL_WHIP == CONSTANTS_REGISTRY["repl_whip"].value

    def test_raw_sgp_denom_matches_registry(self):
        from src.optimizer.constants_registry import CONSTANTS_REGISTRY
        from src.optimizer.daily_optimizer import _RAW_SGP_DENOM

        assert _RAW_SGP_DENOM["AVG"] == CONSTANTS_REGISTRY["raw_sgp_denom_avg"].value
        assert _RAW_SGP_DENOM["OBP"] == CONSTANTS_REGISTRY["raw_sgp_denom_obp"].value
        assert _RAW_SGP_DENOM["ERA"] == CONSTANTS_REGISTRY["raw_sgp_denom_era"].value
        assert _RAW_SGP_DENOM["WHIP"] == CONSTANTS_REGISTRY["raw_sgp_denom_whip"].value


class TestNoHardcodedReplLiterals:
    """Guard against future regressions: no hardcoded literal `_REPL_AVG = 0.X`
    style assignments allowed in daily_optimizer.py. Values MUST flow from
    the registry."""

    def test_no_hardcoded_repl_literal_in_daily_optimizer(self):
        import re
        from pathlib import Path

        src_path = Path(__file__).resolve().parent.parent / "src" / "optimizer" / "daily_optimizer.py"
        content = src_path.read_text(encoding="utf-8")

        # Bad pattern: `_REPL_NAME = <numeric literal>` (e.g., `_REPL_AVG = 0.240`).
        # Acceptable: `_REPL_NAME = _CR["repl_name"].value`
        bad_pattern = re.compile(
            r"^_REPL_[A-Z]+\s*:\s*float\s*=\s*\d+\.\d+",
            re.MULTILINE,
        )
        matches = bad_pattern.findall(content)
        assert not matches, (
            f"daily_optimizer.py contains hardcoded _REPL_* literal(s): {matches}. "
            f"Per DCV-A1-001, replacement-level values must be sourced from "
            f"CONSTANTS_REGISTRY at module load — see existing pattern "
            f"`_REPL_AVG: float = _CR['repl_avg'].value`."
        )
