"""Draft post-draft grade — DB-free service test + route smoke."""

import pandas as pd
from starlette.testclient import TestClient

from api.contracts.draft import DraftConfig, DraftGradeRequest, DraftPick
from api.deps import get_draft_service
from api.main import create_app
from api.services.draft_service import DraftService

_GRADE = {
    "overall_grade": "B+",
    "overall_score": 78.5,
    "team_value_score": 0.8,
    "pick_efficiency_score": 0.7,
    "category_balance_score": 0.65,
    "total_sgp": 42.0,
    "expected_sgp": 40.0,
    "category_projections": {"HR": {"total": 250, "z_score": 1.2}, "SB": {"total": 90, "z_score": -0.5}},
    "strengths": ["HR"],
    "weaknesses": ["SB"],
}


def test_grade_maps_result_and_grades_only_user_picks(monkeypatch):
    import src.draft_grader as dg

    captured = {}

    def _fake_grade(draft_picks, player_pool, config=None):
        captured["picks"] = draft_picks
        return _GRADE

    monkeypatch.setattr(dg, "grade_draft", _fake_grade)

    req = DraftGradeRequest(
        config=DraftConfig(num_teams=12, user_team_index=0),
        pick_log=[
            DraftPick(pick=0, team_index=0, player_id=100, player_name="A", positions="OF"),
            DraftPick(pick=1, team_index=1, player_id=200, player_name="B", positions="SP"),  # opponent → excluded
            DraftPick(pick=23, team_index=0, player_id=300, player_name="C", positions="2B"),
        ],
    )
    resp = DraftService().grade(req, pool=pd.DataFrame([{"player_id": 100}]))
    assert resp.overall_grade == "B+"
    assert resp.overall_score == 78.5
    # ONLY the user's (team_index==0) picks are graded
    assert {p["player_id"] for p in captured["picks"]} == {100, 300}
    assert captured["picks"][0]["pick_number"] == 1  # 1-indexed overall
    assert captured["picks"][1]["round"] == 2  # pick 23 // 12 + 1
    cats = {c.category: c for c in resp.categories}
    assert cats["HR"].z_score == 1.2
    assert resp.strengths == ["HR"]
    assert resp.weaknesses == ["SB"]


def test_grade_graceful_on_failure(monkeypatch):
    import src.draft_grader as dg

    def _boom(*a, **k):
        raise RuntimeError("no pool")

    monkeypatch.setattr(dg, "grade_draft", _boom)
    resp = DraftService().grade(DraftGradeRequest(), pool=object())
    assert resp.overall_grade == "N/A"
    assert resp.categories == []


class _FakeDraftService:
    def grade(self, req):
        from api.contracts.draft import DraftCategoryGrade, DraftGradeResponse

        return DraftGradeResponse(
            overall_grade="A",
            overall_score=90.0,
            categories=[DraftCategoryGrade(category="HR", total=260, z_score=1.5)],
        )


def test_grade_route_returns_grade(monkeypatch):
    # Gate dormant (no STRIPE env) → the endpoint answers without auth.
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    app = create_app()
    app.dependency_overrides[get_draft_service] = lambda: _FakeDraftService()
    resp = TestClient(app).post("/api/draft/grade", json={"config": {}, "pick_log": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_grade"] == "A"
    assert body["categories"][0]["category"] == "HR"


def test_grade_route_documents_pro_gate():
    schema = create_app().openapi()
    assert "/api/draft/grade" in schema["paths"]
    assert "402" in schema["paths"]["/api/draft/grade"]["post"]["responses"]
