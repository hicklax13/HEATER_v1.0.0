import pytest
from starlette.testclient import TestClient

from api.contracts.punt import PuntCategory, PuntResponse
from api.deps import get_punt_service
from api.main import create_app
from api.services.punt_service import PuntService


def test_punt_contract_shape():
    resp = PuntResponse(
        team_name="Team Hickey",
        punt_candidates=["ERA", "WHIP"],
        categories=[
            PuntCategory(cat="ERA", current_rank=11, gainable=False, recommendation="Punt"),
            PuntCategory(cat="HR", current_rank=3, gainable=True, recommendation="Contend"),
        ],
    )
    dumped = resp.model_dump()
    assert dumped["team_name"] == "Team Hickey"
    assert "ERA" in dumped["punt_candidates"]
    assert dumped["categories"][0]["gainable"] is False
    assert dumped["categories"][1]["recommendation"] == "Contend"
    # defaults
    assert PuntResponse(team_name="X").punt_candidates == []
    assert PuntResponse(team_name="X").categories == []
    assert PuntCategory(cat="SB", current_rank=5, gainable=True).recommendation == ""


class _FakePuntService:
    def get_punt(self, team_name: str) -> PuntResponse:
        return PuntResponse(
            team_name=team_name,
            punt_candidates=["ERA"],
            categories=[
                PuntCategory(cat="ERA", current_rank=11, gainable=False, recommendation="Punt"),
            ],
        )


def test_get_punt_returns_contract():
    app = create_app()
    app.dependency_overrides[get_punt_service] = lambda: _FakePuntService()
    client = TestClient(app)
    resp = client.get("/api/punt?team_name=Team+Hickey")
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["punt_candidates"] == ["ERA"]
    assert body["categories"][0]["cat"] == "ERA"


# ---------------------------------------------------------------------------
# Real PuntService mapper tests (DB-free via monkeypatching the engine seam)
# ---------------------------------------------------------------------------


def _fake_analysis(is_punt_cats: dict[str, bool]) -> dict:
    """Build a synthetic category_gap_analysis return value.

    For each cat: if in is_punt_cats with True → rank=11, gainable_positions=0, is_punt=True
                 if False → rank=5, gainable_positions=3, is_punt=False
    """
    analysis = {}
    for cat, should_punt in is_punt_cats.items():
        if should_punt:
            analysis[cat] = {"rank": 11, "gainable_positions": 0, "is_punt": True}
        else:
            analysis[cat] = {"rank": 5, "gainable_positions": 3, "is_punt": False}
    return analysis


def test_punt_service_is_punt_true(monkeypatch):
    """A category with rank>=10 and gainable_positions==0 maps to is_punt=True,
    recommendation='Punt', gainable=False, and appears in punt_candidates."""
    analysis = _fake_analysis({"ERA": True})

    def fake_get_all_team_totals():
        return {"Team Hickey": {"ERA": 4.50}, "Other": {"ERA": 3.80}}

    def fake_category_gap_analysis(your_totals, all_team_totals, your_team_id, config):
        return analysis

    # Monkeypatch at the module-import level inside the service method
    import sys

    # Inject fakes into sys.modules so the lazy imports inside get_punt resolve correctly
    fake_standings = type(sys)("standings_utils")
    fake_standings.get_all_team_totals = fake_get_all_team_totals
    monkeypatch.setitem(sys.modules, "src.standings_utils", fake_standings)

    fake_cat_mod = type(sys)("category_analysis")
    fake_cat_mod.category_gap_analysis = fake_category_gap_analysis
    monkeypatch.setitem(sys.modules, "src.engine.portfolio.category_analysis", fake_cat_mod)

    class _FakeLeagueConfig:
        pass

    fake_val = type(sys)("valuation")
    fake_val.LeagueConfig = _FakeLeagueConfig
    monkeypatch.setitem(sys.modules, "src.valuation", fake_val)

    svc = PuntService()
    result = svc.get_punt("Team Hickey")

    assert result.team_name == "Team Hickey"
    assert "ERA" in result.punt_candidates
    assert len(result.categories) == 1
    cat = result.categories[0]
    assert cat.cat == "ERA"
    assert cat.current_rank == 11
    assert cat.gainable is False
    assert cat.recommendation == "Punt"


def test_punt_service_is_punt_false(monkeypatch):
    """A category with gainable_positions>0 and rank<10 maps to is_punt=False,
    gainable=True, recommendation='Contend', and NOT in punt_candidates."""
    import sys

    analysis = _fake_analysis({"HR": False})

    def fake_get_all_team_totals():
        return {"Team Hickey": {"HR": 42}, "Other": {"HR": 38}}

    def fake_category_gap_analysis(your_totals, all_team_totals, your_team_id, config):
        return analysis

    fake_standings = type(sys)("standings_utils")
    fake_standings.get_all_team_totals = fake_get_all_team_totals
    monkeypatch.setitem(sys.modules, "src.standings_utils", fake_standings)

    fake_cat_mod = type(sys)("category_analysis")
    fake_cat_mod.category_gap_analysis = fake_category_gap_analysis
    monkeypatch.setitem(sys.modules, "src.engine.portfolio.category_analysis", fake_cat_mod)

    class _FakeLeagueConfig:
        pass

    fake_val = type(sys)("valuation")
    fake_val.LeagueConfig = _FakeLeagueConfig
    monkeypatch.setitem(sys.modules, "src.valuation", fake_val)

    svc = PuntService()
    result = svc.get_punt("Team Hickey")

    assert "HR" not in result.punt_candidates
    assert len(result.categories) == 1
    cat = result.categories[0]
    assert cat.cat == "HR"
    assert cat.current_rank == 5
    assert cat.gainable is True
    assert cat.recommendation == "Contend"


def test_punt_service_mixed_categories(monkeypatch):
    """Multiple cats: punt ones go to punt_candidates; contend ones do not."""
    import sys

    analysis = _fake_analysis({"ERA": True, "WHIP": True, "HR": False, "SB": False})

    def fake_get_all_team_totals():
        return {"Team Hickey": {}}

    def fake_category_gap_analysis(your_totals, all_team_totals, your_team_id, config):
        return analysis

    fake_standings = type(sys)("standings_utils")
    fake_standings.get_all_team_totals = fake_get_all_team_totals
    monkeypatch.setitem(sys.modules, "src.standings_utils", fake_standings)

    fake_cat_mod = type(sys)("category_analysis")
    fake_cat_mod.category_gap_analysis = fake_category_gap_analysis
    monkeypatch.setitem(sys.modules, "src.engine.portfolio.category_analysis", fake_cat_mod)

    class _FakeLeagueConfig:
        pass

    fake_val = type(sys)("valuation")
    fake_val.LeagueConfig = _FakeLeagueConfig
    monkeypatch.setitem(sys.modules, "src.valuation", fake_val)

    svc = PuntService()
    result = svc.get_punt("Team Hickey")

    assert set(result.punt_candidates) == {"ERA", "WHIP"}
    recs = {c.cat: c.recommendation for c in result.categories}
    assert recs["ERA"] == "Punt"
    assert recs["WHIP"] == "Punt"
    assert recs["HR"] == "Contend"
    assert recs["SB"] == "Contend"


def test_punt_service_team_not_in_totals(monkeypatch):
    """If the team is absent from all_team_totals, returns empty PuntResponse (graceful)."""
    import sys

    def fake_get_all_team_totals():
        return {"Other Team": {}}

    fake_standings = type(sys)("standings_utils")
    fake_standings.get_all_team_totals = fake_get_all_team_totals
    monkeypatch.setitem(sys.modules, "src.standings_utils", fake_standings)

    fake_cat_mod = type(sys)("category_analysis")
    monkeypatch.setitem(sys.modules, "src.engine.portfolio.category_analysis", fake_cat_mod)

    class _FakeLeagueConfig:
        pass

    fake_val = type(sys)("valuation")
    fake_val.LeagueConfig = _FakeLeagueConfig
    monkeypatch.setitem(sys.modules, "src.valuation", fake_val)

    svc = PuntService()
    result = svc.get_punt("Missing Team")

    assert result.team_name == "Missing Team"
    assert result.categories == []
    assert result.punt_candidates == []


def test_punt_service_engine_raises_degrades_gracefully(monkeypatch):
    """If the engine raises, get_punt swallows it and returns empty contract."""
    import sys

    def fake_get_all_team_totals():
        return {"Team Hickey": {}}

    def fake_category_gap_analysis(**kwargs):
        raise RuntimeError("DB out")

    fake_standings = type(sys)("standings_utils")
    fake_standings.get_all_team_totals = fake_get_all_team_totals
    monkeypatch.setitem(sys.modules, "src.standings_utils", fake_standings)

    fake_cat_mod = type(sys)("category_analysis")
    fake_cat_mod.category_gap_analysis = fake_category_gap_analysis
    monkeypatch.setitem(sys.modules, "src.engine.portfolio.category_analysis", fake_cat_mod)

    class _FakeLeagueConfig:
        pass

    fake_val = type(sys)("valuation")
    fake_val.LeagueConfig = _FakeLeagueConfig
    monkeypatch.setitem(sys.modules, "src.valuation", fake_val)

    svc = PuntService()
    result = svc.get_punt("Team Hickey")

    assert result.team_name == "Team Hickey"
    assert result.categories == []
    assert result.punt_candidates == []


def test_punt_service_hold_recommendation(monkeypatch):
    """A category with gainable_positions==0 but is_punt==False gets 'Hold'."""
    import sys

    # Simulate engine returning a category that is ranked low but engine didn't flag as punt
    analysis = {"W": {"rank": 8, "gainable_positions": 0, "is_punt": False}}

    def fake_get_all_team_totals():
        return {"Team Hickey": {}}

    def fake_category_gap_analysis(your_totals, all_team_totals, your_team_id, config):
        return analysis

    fake_standings = type(sys)("standings_utils")
    fake_standings.get_all_team_totals = fake_get_all_team_totals
    monkeypatch.setitem(sys.modules, "src.standings_utils", fake_standings)

    fake_cat_mod = type(sys)("category_analysis")
    fake_cat_mod.category_gap_analysis = fake_category_gap_analysis
    monkeypatch.setitem(sys.modules, "src.engine.portfolio.category_analysis", fake_cat_mod)

    class _FakeLeagueConfig:
        pass

    fake_val = type(sys)("valuation")
    fake_val.LeagueConfig = _FakeLeagueConfig
    monkeypatch.setitem(sys.modules, "src.valuation", fake_val)

    svc = PuntService()
    result = svc.get_punt("Team Hickey")

    assert "W" not in result.punt_candidates
    assert result.categories[0].recommendation == "Hold"
    assert result.categories[0].gainable is False
