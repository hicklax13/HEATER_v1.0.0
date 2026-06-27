import pathlib

import pytest

from scripts.launch.evidence_registry import (
    VALID_STATUSES,
    load_registry,
    summarize,
    validate,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REGISTRY = _ROOT / "docs" / "launch" / "evidence_registry.yaml"


def _row(**over):
    base = {
        "id": "P0-X",
        "category": "maintainability",
        "phase": 0,
        "description": "desc",
        "status": "passing",
        "subsystem": "api",
        "verify": "pytest -q",
        "metric": "green",
        "evidence": "tests/foo.py",
        "external_review": "none",
        "blocking_ring": "ci",
        "last_verified": "2026-06-26",
        "score_contribution": "maintainability",
    }
    base.update(over)
    return base


def test_valid_registry_has_no_errors():
    assert validate({"requirements": [_row()]}) == []


def test_missing_required_field_is_reported():
    bad = _row()
    del bad["status"]
    errors = validate({"requirements": [bad]})
    assert any("status" in e for e in errors)


def test_invalid_status_is_reported():
    errors = validate({"requirements": [_row(status="done")]})
    assert any("status" in e and "done" in e for e in errors)


def test_duplicate_ids_are_reported():
    errors = validate({"requirements": [_row(id="DUP"), _row(id="DUP")]})
    assert any("DUP" in e for e in errors)


def test_summarize_counts_by_status():
    reg = {"requirements": [_row(status="passing"), _row(id="P0-Y", status="planned")]}
    summary = summarize(reg)
    assert summary["passing"] == 1
    assert summary["planned"] == 1


def test_status_enum_is_the_documented_set():
    assert VALID_STATUSES == {
        "planned",
        "in_progress",
        "passing",
        "failing",
        "deferred",
        "waived",
    }


@pytest.mark.skipif(not _REGISTRY.exists(), reason="registry not created yet")
def test_committed_registry_validates():
    assert validate(load_registry(_REGISTRY)) == []
