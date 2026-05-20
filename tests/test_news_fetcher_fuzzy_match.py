"""Test the optimized fuzzy-match path in src/news_fetcher.py.

PR 2026-05-19: replaced O(N*M) brute-force SequenceMatcher loop with a
3-tier lookup that uses canonical normalize_player_name as a fast path.

The original impl ran SequenceMatcher.ratio() against EVERY candidate
on every miss, causing CI shard 1 to hang 20+ min when the candidate
pool was large. The optimized impl resolves ~98% of cases via exact
lookups (case-insensitive + canonical-normalized), only falls back to
SequenceMatcher with a prefix filter that bounds the inner loop.

API stays backward-compatible: `_fuzzy_match_name(name, candidates_lower, threshold)`
still accepts a flat lowercase-keyed dict. An optional
`canonical_candidates` kwarg lets callers (like aggregate_player_news)
pre-build the canonical lookup once, avoiding O(N) rebuild on every
query.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from src.news_fetcher import _fuzzy_match_name, aggregate_player_news
from src.valuation import normalize_player_name


@contextmanager
def _capture_logs(logger_name: str, level: int = logging.DEBUG):
    """Capture log messages emitted by ``logger_name`` at or above ``level``."""
    captured: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    logger = logging.getLogger(logger_name)
    handler = _Handler(level=level)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield captured
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _lower(pairs: list[tuple[str, int]]) -> dict[str, int]:
    return {name.lower().strip(): pid for name, pid in pairs}


def _canonical(pairs: list[tuple[str, int]]) -> dict[str, int]:
    return {normalize_player_name(name): pid for name, pid in pairs if normalize_player_name(name)}


# ── Direct _fuzzy_match_name tests ─────────────────────────────────────


def test_exact_lowercase_match():
    """Pass 1: exact case-insensitive hit returns immediately."""
    cands = _lower([("Aaron Judge", 1)])
    assert _fuzzy_match_name("aaron judge", cands) == 1
    assert _fuzzy_match_name("AARON JUDGE", cands) == 1
    assert _fuzzy_match_name("Aaron Judge", cands) == 1


def test_canonical_normalize_match_accents():
    """Pass 2: 'Iván Rodríguez' resolves via canonical normalize when DB has 'Ivan Rodriguez'."""
    cands = _lower([("Ivan Rodriguez", 42)])
    assert _fuzzy_match_name("Iván Rodríguez", cands) == 42


def test_canonical_normalize_match_suffix():
    """Pass 2: 'Vladimir Guerrero Jr.' resolves when DB has 'Vladimir Guerrero'."""
    cands = _lower([("Vladimir Guerrero", 27)])
    assert _fuzzy_match_name("Vladimir Guerrero Jr.", cands) == 27


def test_canonical_normalize_match_yahoo_parenthetical():
    """Pass 2: 'Shohei Ohtani (Pitcher)' resolves to 'Shohei Ohtani'."""
    cands = _lower([("Shohei Ohtani", 17)])
    assert _fuzzy_match_name("Shohei Ohtani (Pitcher)", cands) == 17


def test_sequencematcher_fallback_for_typos():
    """Pass 3: SequenceMatcher catches small typos canonical normalize can't (last-char drop)."""
    cands = _lower([("Aaron Judge", 1)])
    assert _fuzzy_match_name("Aaron Judg", cands) == 1


def test_no_match_below_threshold():
    cands = _lower([("Aaron Judge", 1)])
    assert _fuzzy_match_name("Mike Trout", cands) is None


def test_empty_candidates_returns_none():
    assert _fuzzy_match_name("Anyone", {}) is None


def test_empty_name_returns_none():
    cands = _lower([("Anyone", 1)])
    assert _fuzzy_match_name("", cands) is None


def test_pre_built_canonical_candidates_kwarg():
    """Caller-supplied canonical_candidates avoids rebuilding on every query."""
    pairs = [("Ivan Rodriguez", 42), ("Vladimir Guerrero", 27)]
    cands_lower = _lower(pairs)
    cands_canonical = _canonical(pairs)
    assert _fuzzy_match_name("Iván Rodríguez", cands_lower, canonical_candidates=cands_canonical) == 42
    assert _fuzzy_match_name("Vladimir Guerrero Jr.", cands_lower, canonical_candidates=cands_canonical) == 27


def test_prefix_pruning_speed():
    """With 9000 candidates, fallback should still complete <2.5s on typo.

    2026-05-20 SFH J: bumped fuzzy threshold from 1s → 2.5s and exact from
    50ms → 200ms. GitHub Actions free-tier runners are noticeably slower
    than local dev machines (observed 1.2s for fuzzy on CI vs ~0.3s
    locally). The original tight thresholds were what flagged Coverage
    Floor red on every PR since #54 — the failure was misattributed to
    coverage because both run in the same job. The asserts still serve
    the original purpose (catch a hung-loop regression that brought back
    the 20+ min CI shard 1 hang) just with more CI headroom.
    """
    pairs = [(f"Player{i:04d} LastName{i:04d}", i) for i in range(9000)]
    cands_lower = _lower(pairs)
    cands_canonical = _canonical(pairs)

    # Exact match — should be near-instant (dict lookup) even on slow CI.
    t0 = time.monotonic()
    result = _fuzzy_match_name("Player0500 LastName0500", cands_lower, canonical_candidates=cands_canonical)
    elapsed_exact = time.monotonic() - t0
    assert result == 500
    assert elapsed_exact < 0.20, f"Exact match should be <200ms; got {elapsed_exact * 1000:.1f}ms"

    # Typo match — must use SequenceMatcher fallback but with prefix pruning.
    # "Player0500 LastNme0500" drops a letter mid-word; canonical normalize
    # can't fix it. The 2.5s cap is well below the 20-minute hang we're
    # guarding against, while still tolerating CI variability.
    t0 = time.monotonic()
    _fuzzy_match_name("Player0500 LastNme0500", cands_lower, canonical_candidates=cands_canonical)
    elapsed_fuzzy = time.monotonic() - t0
    assert elapsed_fuzzy < 2.5, f"Fuzzy fallback should be <2.5s with prefix pruning; got {elapsed_fuzzy * 1000:.1f}ms"


# ── aggregate_player_news end-to-end ──────────────────────────────────


def test_aggregate_player_news_smoke():
    """End-to-end: transactions + name->id dict → grouped news, with canonical accent rescue."""
    transactions = [
        {"player_name": "Aaron Judge", "description": "Injured", "date": "2026-05-19", "transaction_type": "IL"},
        {"player_name": "Iván Rodríguez", "description": "Trade", "date": "2026-05-19", "transaction_type": "Trade"},
        {"player_name": "Unknown Player", "description": "Skip", "date": "2026-05-19", "transaction_type": "Skip"},
    ]
    name_to_id = {"Aaron Judge": 1, "Ivan Rodriguez": 42}

    result = aggregate_player_news(transactions, name_to_id)
    assert result[1] == ["Injured"]
    assert result[42] == ["Trade"]  # Matched via canonical normalize (accent strip)
    assert 999 not in result


def test_aggregate_player_news_handles_empty():
    assert aggregate_player_news([], {}) == {}
    assert aggregate_player_news([{"player_name": "X", "description": "Y"}], {}) == {}
    assert aggregate_player_news([], {"X": 1}) == {}


def test_accented_query_typo_finds_unaccented_candidate():
    """SFH H-1 regression: accented query with a typo should still match
    via canonical-prefix prune in Pass 3.

    Pre-fix: prune by `name_lower[:2]` meant `"Étiene"` (typo of Étienne)
    prefix was "ét" while candidate "Etienne"[:2] was "et" — silently
    skipped. Post-fix: prune by canonical prefix so both become "et".
    """
    cands = _lower([("Etienne Bernier", 77)])
    cands_canonical = _canonical([("Etienne Bernier", 77)])
    # "Étiene" is a 1-letter typo of "Etienne" with an accent — canonical
    # normalize won't make them identical, so this MUST exercise Pass 3.
    result = _fuzzy_match_name("Étiene Bernier", cands, canonical_candidates=cands_canonical)
    assert result == 77, "Accented typo should fuzzy-match via canonical prune"


def test_no_kwarg_path_logs_debug():
    """SFH H-2 regression: when canonical_candidates kwarg is None and we
    have to rebuild on the fly, emit a debug log so hot-loop misuse is
    visible to ops.

    Query must MISS Pass 1 (exact case-insensitive) but HIT Pass 2 — that's
    the only path where the rebuild happens. "Iván Rodríguez" → exact match
    in lower lookup; "Ivan Rodriguez" (no accent) misses lower (because lower
    has accent) and only resolves via canonical.
    """
    # Lower lookup has accented form; query is unaccented → forces Pass 2.
    cands = {"iván rodríguez": 42}
    with _capture_logs("src.news_fetcher", logging.DEBUG) as captured:
        result = _fuzzy_match_name("Ivan Rodriguez", cands)
    assert result == 42
    assert any("rebuilding canonical_candidates" in msg for msg in captured), (
        f"Expected debug log about rebuilding; got: {captured}"
    )


def test_pass3_uses_prebuilt_canonical_candidates():
    """SFH M/H regression: Pass 3 iterates over canonical_candidates (not
    candidates), so a typo-heavy query with 9000 candidates does NOT
    recompute normalize_player_name on every candidate.

    Pre-PR #58: Pass 3 looped over `candidates.items()` and called
    `normalize_player_name(cand_name)` inside the loop — pre-built
    canonical_candidates only helped Pass 2 (exact lookup).
    Post-PR #58: Pass 3 iterates `canonical_candidates.items()` directly,
    eliminating the per-candidate normalize() call.

    This test counts normalize_player_name invocations during a Pass 3
    lookup; expects ZERO calls per candidate (only the 1 call to canonicalize
    the query itself + 0 for candidates).
    """
    from unittest.mock import patch

    import src.news_fetcher as news_fetcher

    pairs = [(f"Player{i:04d} Name{i:04d}", i) for i in range(100)]
    cands_lower = _lower(pairs)
    cands_canonical = _canonical(pairs)

    call_count = {"n": 0}
    real_normalize = news_fetcher.normalize_player_name

    def _counting_normalize(name):
        call_count["n"] += 1
        return real_normalize(name)

    with patch.object(news_fetcher, "normalize_player_name", side_effect=_counting_normalize):
        # Query that misses Pass 1 + 2 (forces Pass 3): drop a letter to make a typo.
        result = _fuzzy_match_name(
            "Player0050 Nme0050",
            cands_lower,
            canonical_candidates=cands_canonical,
        )

    # 1 normalize for the query itself; Pass 3 should NOT recompute for any of 100 candidates.
    assert call_count["n"] <= 3, (
        f"Pass 3 should not recompute normalize per candidate when canonical_candidates "
        f"is pre-built. Got {call_count['n']} normalize calls for 100 candidates "
        f"(expected ≤3: query canonical + maybe canonical-rebuild branches)."
    )
    assert result == 50, "Pass 3 should still find the typo'd candidate"


def test_aggregate_at_scale_completes_under_5s():
    """50 transactions × 9000 candidates: must complete in <5s.

    Pre-PR: this scenario hung indefinitely (CI shard 1 ran 20+ min).
    Post-PR: <5s on a typical CI runner.
    """
    candidates_data = [(f"Player{i:04d} Name{i:04d}", i) for i in range(9000)]
    transactions = [
        {
            "player_name": f"Player{i * 50:04d} Name{i * 50:04d}",
            "description": f"News {i}",
            "date": "2026-05-19",
            "transaction_type": "Note",
        }
        for i in range(50)
    ]
    name_to_id = dict(candidates_data)

    t0 = time.monotonic()
    result = aggregate_player_news(transactions, name_to_id)
    elapsed = time.monotonic() - t0

    assert len(result) == 50, f"Expected 50 matches; got {len(result)}"
    assert elapsed < 5.0, f"Aggregate-at-scale should be <5s; got {elapsed:.2f}s"
