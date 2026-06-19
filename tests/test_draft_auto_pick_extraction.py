"""Behavior-identical guard for the slice-9 extraction of the Draft Simulator's
AI-opponent picking out of pages/20_Draft_Simulator.py into src/simulation.py.

The frozen reference below is a verbatim copy of the OLD inline page logic
(pre-extraction). The extracted engine fn must produce the identical pick
sequence under the same RNG state, so the live page (now a thin delegate)
keeps byte-identical behavior."""

import ast
import pathlib

import numpy as np
import pandas as pd

from src.draft_state import DraftState
from src.simulation import auto_pick_opponents

_PAGE = pathlib.Path(__file__).resolve().parents[1] / "pages" / "20_Draft_Simulator.py"


def _pool() -> pd.DataFrame:
    positions = ["SS", "OF", "2B", "3B", "1B", "SP", "RP", "C"]
    rows = [
        {
            "player_id": 100 + i,
            "name": f"Player{i}",
            "player_name": f"Player{i}",
            "positions": positions[i % len(positions)],
            "adp": float(i + 1),
        }
        for i in range(20)
    ]
    return pd.DataFrame(rows)


def _state(user_seat: int = 2, num_teams: int = 12, num_rounds: int = 23) -> DraftState:
    return DraftState(num_teams=num_teams, num_rounds=num_rounds, user_team_index=user_seat)


def _reference_auto_pick(ds: DraftState, pool: pd.DataFrame) -> list[int]:
    """Frozen copy of the OLD pages/20_Draft_Simulator.py inline logic."""
    made: list[int] = []
    while not ds.is_user_turn and ds.current_pick < ds.total_picks:
        available = ds.available_players(pool)
        if available.empty:
            break
        candidates = available.nsmallest(min(15, len(available)), "adp")
        size = len(candidates)
        weights = np.arange(size, 0, -1, dtype=float)
        weights /= weights.sum()
        pick_idx = int(np.random.choice(size, p=weights))
        player = candidates.iloc[pick_idx]
        pname = str(player.get("player_name", player.get("name", "Unknown")))
        made.append(int(player["player_id"]))
        ds.make_pick(
            player_id=int(player["player_id"]),
            player_name=pname,
            positions=str(player.get("positions", "Util")),
        )
    return made


def test_extracted_matches_frozen_reference():
    pool = _pool()
    np.random.seed(20260619)
    ref_ids = _reference_auto_pick(_state(), pool)
    np.random.seed(20260619)
    new_ids = [p["player_id"] for p in auto_pick_opponents(_state(), pool)]
    assert new_ids == ref_ids
    assert len(ref_ids) == 2  # user at seat 2 → opponents fill seats 0,1 then stop


def test_seeded_rng_is_deterministic():
    pool = _pool()
    a = [p["player_id"] for p in auto_pick_opponents(_state(), pool, rng=np.random.default_rng(123))]
    b = [p["player_id"] for p in auto_pick_opponents(_state(), pool, rng=np.random.default_rng(123))]
    assert a == b and len(a) == 2


def test_returns_pick_metadata():
    made = auto_pick_opponents(_state(user_seat=2), _pool(), rng=np.random.default_rng(1))
    assert [m["team_index"] for m in made] == [0, 1]
    assert [m["pick"] for m in made] == [0, 1]
    for m in made:
        assert set(m) == {"pick", "team_index", "player_id", "player_name", "positions"}


def test_no_picks_when_user_already_on_clock():
    made = auto_pick_opponents(_state(user_seat=0), _pool(), rng=np.random.default_rng(1))
    assert made == []


def test_page_delegates_to_engine_no_inline_loop():
    """The page's auto_pick_opponents must import + call the engine fn and
    contain no inline RNG pick (so it can never drift from the engine)."""
    tree = ast.parse(_PAGE.read_text(encoding="utf-8"))

    asname = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.simulation":
            for alias in node.names:
                if alias.name == "auto_pick_opponents":
                    asname = alias.asname or alias.name
    assert asname, "page must import auto_pick_opponents from src.simulation"

    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "auto_pick_opponents")
    # inline MC pick is gone — no `.choice` call remains in the page fn
    assert not any(isinstance(n, ast.Attribute) and n.attr == "choice" for n in ast.walk(func)), (
        "page auto_pick_opponents still contains an inline .choice call"
    )
    # it delegates to the imported engine fn
    called = {n.func.id for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert asname in called, "page auto_pick_opponents must call the src.simulation delegate"
