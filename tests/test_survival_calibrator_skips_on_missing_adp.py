"""BUG-015 fix: survival_calibrator skips with clear warning when no real ADP plumbed."""

import logging

import pandas as pd

from src.validation.calibration_data import CalibrationDataset, DraftPick


def _make_dataset(season: int, n_picks: int = 10, num_teams: int = 10) -> CalibrationDataset:
    """Build a minimal CalibrationDataset with no real ADP plumbed."""
    ds = CalibrationDataset(
        league_key=f"469.l.test.{season}",
        season=season,
        num_teams=num_teams,
    )
    ds.draft_picks = [
        DraftPick(
            pick_number=i,
            round=((i - 1) // num_teams) + 1,
            team_key=f"team_{((i - 1) % num_teams) + 1}",
            team_name=f"Team {((i - 1) % num_teams) + 1}",
            player_name=f"Player{i}",
            player_id=1000 + i,
        )
        for i in range(1, n_picks + 1)
    ]
    return ds


class _DatasetWithRealAdp(CalibrationDataset):
    """Subclass that surfaces a real `adp` column in to_draft_dataframe()."""

    adp_overrides: dict[int, float]

    def to_draft_dataframe(self) -> pd.DataFrame:
        df = super().to_draft_dataframe()
        if df.empty:
            return df
        df["adp"] = df["pick_number"].map(self.adp_overrides)
        return df


def test_calibrator_does_not_use_actual_pick_as_adp(caplog):
    """The survival calibrator must NOT silently use actual_pick as adp
    (data leakage). It should either accept a real adp column or skip
    with a clear warning."""
    from src.validation.survival_calibrator import _build_prediction_pairs

    ds = _make_dataset(season=2026, n_picks=10, num_teams=10)

    with caplog.at_level(logging.WARNING, logger="src.validation.survival_calibrator"):
        pairs = _build_prediction_pairs([ds], num_teams=10)

    # Either pairs is empty (skipped) OR pairs don't have adp == actual_pick
    if pairs:
        adp_equals_actual_count = sum(
            1
            for p in pairs
            if p.get("adp") is not None and p.get("actual_pick") is not None and p["adp"] == p["actual_pick"]
        )
        assert adp_equals_actual_count == 0, (
            f"BUG-015 regression: {adp_equals_actual_count}/{len(pairs)} survival "
            f"pairs have adp == actual_pick (data leakage). When no real ADP is "
            f"plumbed, the calibrator should SKIP rather than fabricate the feature."
        )

    # If we skipped, expect a warning log
    if not pairs:
        log_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        warned = any("ADP" in m.upper() or "skip" in m.lower() for m in log_messages)
        assert warned, (
            "BUG-015: when survival_calibrator skips due to missing ADP, it must "
            f"emit a WARNING log explaining why. Got log records: {log_messages}"
        )


def test_calibrator_uses_real_adp_when_present():
    """When a real `adp` column IS in the draft df, the calibrator should
    use it (NOT actual_pick)."""
    from src.validation.survival_calibrator import _build_prediction_pairs

    ds = _DatasetWithRealAdp(
        league_key="469.l.test.2026",
        season=2026,
        num_teams=10,
    )
    ds.draft_picks = [
        DraftPick(
            pick_number=i,
            round=1,
            team_key=f"team_{i}",
            team_name=f"Team {i}",
            player_name=f"Player{chr(64 + i)}",
            player_id=1000 + i,
        )
        for i in range(1, 4)
    ]
    # Real ADP differs from actual_pick (drafted earlier than ADP suggested)
    ds.adp_overrides = {1: 5.0, 2: 10.0, 3: 15.0}

    pairs = _build_prediction_pairs([ds], num_teams=10)

    assert pairs, (
        "BUG-015: with a real `adp` column present, the calibrator should "
        "produce prediction pairs (not skip the dataset)."
    )

    # At least some pair should have adp != actual_pick (the real ADP differs)
    non_leakage = sum(
        1
        for p in pairs
        if p.get("adp") is not None and p.get("actual_pick") is not None and float(p["adp"]) != float(p["actual_pick"])
    )
    assert non_leakage > 0, (
        "BUG-015: even with real ADP column present, calibrator silently "
        "overrode adp = actual_pick. All pairs leak the target."
    )
