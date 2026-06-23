"""Matchup-C — the league scoreboard (`league[]`). DB-free: engines monkeypatched
at their source modules (the service imports them lazily inside methods; the
worktree/CI DB is empty — see the reference_worktree_empty_db memory)."""

from __future__ import annotations

from api.contracts.matchup import LeagueMatchup, MatchupResponse, TeamSide
from api.services.matchup_service import MatchupService


def _svc():
    return MatchupService()


def _cats(pairs):
    """[(cat, you, opp), ...] → the raw category-dict list _build_categories expects."""
    return [{"cat": c, "you": str(y), "opp": str(o)} for c, y, o in pairs]


def test_league_builds_all_pairings_user_and_cached_scores(monkeypatch):
    # week 13: the user's pairing + one other pairing
    schedule = {13: [("Team Hickey", "Baty Babies"), ("BUBBA", "Cyrus")]}
    monkeypatch.setattr("src.database.load_league_schedule_full", lambda: schedule)

    # BUBBA's cached matchup vs Cyrus: BUBBA wins HR + ERA(inverse), loses R, ties SB → (2 wins, 1 loss)
    def fake_cache(name, week):
        if name == "BUBBA":
            return {"categories": _cats([("HR", 5, 3), ("R", 4, 6), ("SB", 2, 2), ("ERA", 3.0, 4.0)])}
        return None

    monkeypatch.setattr("src.database.load_matchup_cache", fake_cache)

    league = _svc()._league("Team Hickey", "Baty Babies", week=13, you_score=7, opp_score=5)
    assert len(league) == 2

    # entry 0 = the user's own pairing → reuses the header you/opp scores, aligned a/b
    assert league[0].a.name == "Team Hickey" and league[0].a.score == 7
    assert league[0].b.name == "Baty Babies" and league[0].b.score == 5

    # entry 1 = other pairing → scores from BUBBA's cache (2 wins, 1 loss → Cyrus 1)
    assert league[1].a.name == "BUBBA" and league[1].a.score == 2
    assert league[1].b.name == "Cyrus" and league[1].b.score == 1


def test_league_user_pairing_aligns_when_user_is_team_b(monkeypatch):
    # schedule lists the user as team_b — scores must still align (opp=a, you=b)
    monkeypatch.setattr("src.database.load_league_schedule_full", lambda: {13: [("Baty Babies", "Team Hickey")]})
    monkeypatch.setattr("src.database.load_matchup_cache", lambda n, w: None)
    league = _svc()._league("Team Hickey", "Baty Babies", week=13, you_score=7, opp_score=5)
    assert league[0].a.name == "Baty Babies" and league[0].a.score == 5  # opponent
    assert league[0].b.name == "Team Hickey" and league[0].b.score == 7  # you


def test_league_uncached_other_pairing_scores_zero(monkeypatch):
    monkeypatch.setattr("src.database.load_league_schedule_full", lambda: {13: [("BUBBA", "Cyrus")]})
    monkeypatch.setattr("src.database.load_matchup_cache", lambda n, w: None)  # nothing cached
    league = _svc()._league("Team Hickey", "Baty Babies", week=13, you_score=7, opp_score=5)
    assert len(league) == 1
    assert league[0].a.score == 0 and league[0].b.score == 0  # pairing shown, scores degrade to 0


def test_league_cold_env_returns_empty(monkeypatch):
    # week 0 → []
    assert _svc()._league("Team Hickey", "Baty", week=0, you_score=0, opp_score=0) == []

    # schedule source raising → [] (never propagates)
    def boom():
        raise RuntimeError("no schedule table")

    monkeypatch.setattr("src.database.load_league_schedule_full", boom)
    assert _svc()._league("Team Hickey", "Baty", week=13, you_score=7, opp_score=5) == []


def test_score_from_cache_counts_wins_losses(monkeypatch):
    monkeypatch.setattr(
        "src.database.load_matchup_cache",
        lambda n, w: {"categories": _cats([("HR", 5, 3), ("R", 4, 6), ("WHIP", 1.0, 1.2)])},
    )
    # HR you-win, R opp-win, WHIP(inverse) you-win → (2 wins, 1 loss)
    assert _svc()._score_from_cache("BUBBA", 13) == (2, 1)


def test_score_from_cache_none_when_uncached(monkeypatch):
    monkeypatch.setattr("src.database.load_matchup_cache", lambda n, w: None)
    assert _svc()._score_from_cache("BUBBA", 13) is None


def test_pairing_scores_uses_b_cache_when_a_missing(monkeypatch):
    # Only team b (Cyrus) is cached; team a (BUBBA) is not → the swap path.
    def fake_cache(name, week):
        if name == "Cyrus":  # Cyrus wins R, loses HR → from Cyrus's view (1 win, 1 loss)
            return {"categories": _cats([("HR", 3, 5), ("R", 6, 4)])}
        return None

    monkeypatch.setattr("src.database.load_matchup_cache", fake_cache)
    # (a=BUBBA, b=Cyrus): a-cache miss → b-cache (1,1) swapped to (a, b) = (1, 1)
    assert _svc()._pairing_scores("BUBBA", "Cyrus", 13) == (1, 1)


def test_pairing_scores_swap_preserves_order(monkeypatch):
    # Asymmetric counts prove the swap maps b's (b_wins, a_wins) back to (a_wins, b_wins).
    def fake_cache(name, week):
        if name == "Cyrus":  # Cyrus wins HR+R (2), loses SB (1) → from Cyrus's view (2, 1)
            return {"categories": _cats([("HR", 5, 3), ("R", 6, 4), ("SB", 1, 2)])}
        return None

    monkeypatch.setattr("src.database.load_matchup_cache", fake_cache)
    # (a=BUBBA, b=Cyrus): b-cache (2,1) → swap → a=BUBBA gets 1, b=Cyrus gets 2
    assert _svc()._pairing_scores("BUBBA", "Cyrus", 13) == (1, 2)


def test_league_contract_round_trips():
    resp = MatchupResponse(
        team_name="Team Hickey",
        league=[
            LeagueMatchup(
                a=TeamSide(name="Team Hickey", manager="Connor", record="4-7-1 · 8th", score=7),
                b=TeamSide(name="Baty Babies", manager="dandre", record="6-4-2 · 4th", score=5),
            )
        ],
    )
    dumped = resp.model_dump()
    assert dumped["league"][0]["a"]["name"] == "Team Hickey"
    assert dumped["league"][0]["a"]["score"] == 7
    assert dumped["league"][0]["b"]["manager"] == "dandre"


def test_build_categories_preserves_negative_and_dash_placeholder():
    """Yahoo's '-' (not-yet-played) → 0, but a real negative value KEEPS its sign.
    The old `str(x).replace('-','0')` turned '-5' into '05'=5.0 (sign flip) — masked
    only because H2H category totals are normally non-negative."""
    raw = [
        {"cat": "R", "you": "-5", "opp": "3"},
        {"cat": "HR", "you": "-", "opp": "2"},  # Yahoo not-yet-played placeholder
        {"cat": "AVG", "you": "0.275", "opp": "0.260"},
    ]
    by = {c.cat: c for c in MatchupService._build_categories(raw, "Team A")}
    assert by["R"].you == -5.0  # NOT 5.0
    assert by["HR"].you == 0.0  # placeholder → 0
    assert by["AVG"].you == 0.275
