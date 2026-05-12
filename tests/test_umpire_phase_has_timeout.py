"""INFRA-F3 fix: _bootstrap_umpire_tendencies must have a wall-clock timeout
on Tier 1 (schedule iteration) so it falls through to Tier 3 seed in bounded time."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_umpire_phase_uses_time_budget():
    """The Tier 1 schedule iteration in _bootstrap_umpire_tendencies must use
    a `time.time()` budget so it terminates within ~60s even with thousands
    of completed games."""
    text = (REPO_ROOT / "src" / "data_bootstrap.py").read_text(encoding="utf-8")

    start = text.find("def _bootstrap_umpire_tendencies")
    assert start >= 0, "umpire phase function not found"
    end = text.find("\ndef ", start + 1)
    body = text[start : end if end > 0 else len(text)]

    has_budget = ("time.time()" in body or "time.monotonic()" in body) and (
        "UMPIRE_TIER1_TIMEOUT" in body or "_TIER1_TIMEOUT" in body or "elapsed" in body.lower()
    )
    assert has_budget, (
        "INFRA-F3 regression: _bootstrap_umpire_tendencies does not appear to "
        "use a wall-clock timeout to cap Tier 1 schedule iteration. The phase "
        "should track elapsed time and break to Tier 3 seed when exceeded. "
        f"Body excerpt:\n{body[:500]}"
    )
