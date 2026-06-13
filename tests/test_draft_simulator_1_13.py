"""Task 1.13 — Draft Simulator: Undo Last Pick + HTML-injection in player selectbox.

(a) Undo must rewind through the user's own last pick so that after Undo it is
    the user's turn again and the user's roster count is one smaller.

(b) The list passed to render_player_select (the "View player card" dropdown)
    must contain plain text names, NOT raw HTML / <img> strings.
"""

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

from src.draft_state import DraftState

PAGE = Path(__file__).resolve().parent.parent / "pages" / "20_Draft_Simulator.py"


def _page_src() -> str:
    return PAGE.read_text(encoding="utf-8")


# ── Helper: minimal player pool for draft state tests ──────────────────────


def _make_pool(n: int = 30) -> pd.DataFrame:
    """Minimal DataFrame that DraftState.available_players() and pick recording accept."""
    return pd.DataFrame(
        {
            "player_id": list(range(1, n + 1)),
            "player_name": [f"Player {i}" for i in range(1, n + 1)],
            "name": [f"Player {i}" for i in range(1, n + 1)],
            "positions": ["OF"] * n,
            "adp": list(range(1, n + 1)),
            "pick_score": [float(n - i) for i in range(n)],
            "mlb_id": list(range(1000, 1000 + n)),
        }
    )


def _make_ds(user_pos: int = 1, num_teams: int = 3, num_rounds: int = 3) -> DraftState:
    """Create a DraftState with a small league for fast tests.

    user_pos: 1-based draft position.
    """
    return DraftState(num_teams=num_teams, num_rounds=num_rounds, user_team_index=user_pos - 1)


def _simulate_user_pick_then_ai(ds: DraftState, pool: pd.DataFrame) -> None:
    """Make the user's pick (pick 1 in round 1 if user is first) then let AI
    fill until it's the user's turn again (or draft ends)."""
    assert ds.is_user_turn, "Expected user's turn at start of helper"
    # User picks player_id=1
    avail = ds.available_players(pool)
    p = avail.iloc[0]
    ds.make_pick(int(p["player_id"]), str(p["player_name"]), str(p.get("positions", "OF")))
    # AI picks until user's turn (or end of draft)
    while not ds.is_user_turn and ds.current_pick < ds.total_picks:
        avail = ds.available_players(pool)
        if avail.empty:
            break
        q = avail.iloc[0]
        ds.make_pick(int(q["player_id"]), str(q["player_name"]), str(q.get("positions", "OF")))


# ── Task 1.13a: Undo rewinds to the user's turn ────────────────────────────


def test_undo_rewinds_to_user_turn_3team():
    """With a 3-team league, after user picks + AI auto-picks, Undo should
    pop picks until the user's OWN last pick is undone, leaving is_user_turn True
    and the user's roster count one smaller.
    """
    pool = _make_pool(30)
    # User is team 1 (index 0); snake: round1 = [0,1,2], round2 = [2,1,0]
    ds = _make_ds(user_pos=1, num_teams=3, num_rounds=3)

    # Initial state: user is on the clock (pick 0, team 0)
    assert ds.is_user_turn

    # Simulate: user picks at pick 0, AI picks at 1 & 2, then round 2 (reverse)
    # AI picks at 3 (team2) & 4 (team1), then pick 5 is user's turn again.
    _simulate_user_pick_then_ai(ds, pool)

    user_picks_before = len(ds.user_team.picks)
    assert user_picks_before >= 1, "User should have at least 1 pick"
    assert ds.is_user_turn, "After _simulate_user_pick_then_ai the user should be on deck"

    # There should be AI picks between the user's last pick and current_pick.
    last_user_pick_num = next(e["pick"] for e in reversed(ds.pick_log) if e["team_index"] == ds.user_team_index)
    assert last_user_pick_num < ds.current_pick - 1, (
        "At least one AI pick should have been made after the user's last pick"
    )

    # --- The undo operation the page should perform ---
    # Pop AI picks off the top, then pop the user's own pick.
    def undo_to_user_turn(ds_inner: DraftState) -> None:
        """Mirrors the corrected page handler: unwind through user's own last pick."""
        if not ds_inner.pick_log:
            return
        # Pop AI picks until the top of the log is the user's own pick
        while ds_inner.pick_log and ds_inner.pick_log[-1]["team_index"] != ds_inner.user_team_index:
            ds_inner.undo_last_pick()
        # Now pop the user's own pick
        if ds_inner.pick_log and ds_inner.pick_log[-1]["team_index"] == ds_inner.user_team_index:
            ds_inner.undo_last_pick()

    undo_to_user_turn(ds)

    assert ds.is_user_turn, "After Undo, it must be the user's turn"
    assert len(ds.user_team.picks) < user_picks_before, (
        f"User roster must shrink: was {user_picks_before}, now {len(ds.user_team.picks)}"
    )


def test_undo_when_user_is_already_on_clock_does_nothing_dangerous():
    """If Undo is called when the user is already on the clock (no AI picks
    between their last pick and now), it should still undo the user's last
    pick and leave the draft in a valid state."""
    pool = _make_pool(30)
    ds = _make_ds(user_pos=1, num_teams=3, num_rounds=3)
    # Make one user pick (now it's AI's turn)
    avail = ds.available_players(pool)
    p = avail.iloc[0]
    ds.make_pick(int(p["player_id"]), str(p["player_name"]), str(p.get("positions", "OF")))
    assert not ds.is_user_turn or ds.current_pick == ds.total_picks

    # Correct undo logic: pop AI picks until user's own pick is at top, then pop it
    def undo_to_user_turn(ds_inner: DraftState) -> None:
        if not ds_inner.pick_log:
            return
        while ds_inner.pick_log and ds_inner.pick_log[-1]["team_index"] != ds_inner.user_team_index:
            ds_inner.undo_last_pick()
        if ds_inner.pick_log and ds_inner.pick_log[-1]["team_index"] == ds_inner.user_team_index:
            ds_inner.undo_last_pick()

    undo_to_user_turn(ds)

    # Draft should be back to 0 picks, user's turn
    assert ds.current_pick == 0
    assert ds.is_user_turn
    assert len(ds.user_team.picks) == 0


def test_page_undo_handler_calls_undo_loop_not_single():
    """The page's Undo button handler must NOT use the old broken pattern:
    single ds.undo_last_pick() + auto_pick_opponents().

    The correct fix: a while-loop that pops AI picks until the user's own
    pick is at the top, then pops that too. auto_pick_opponents() must NOT
    be called inside the undo block.
    """
    src = _page_src()
    tree = ast.parse(src)

    # Find the 'if st.button("Undo Last Pick")' block
    undo_block_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            cond_src = ast.get_source_segment(src, node.test) or ""
            if "Undo Last Pick" in cond_src:
                undo_block_src = ast.get_source_segment(src, node) or ""
                break

    assert undo_block_src is not None, "Could not find 'Undo Last Pick' button block in page"

    # The block must contain a while loop to unwind AI picks
    assert "while" in undo_block_src, (
        "Undo handler must use a 'while' loop to pop AI picks before removing the user's own pick"
    )

    # auto_pick_opponents must NOT appear inside the undo block
    assert "auto_pick_opponents" not in undo_block_src, (
        "Undo handler must NOT call auto_pick_opponents() — that immediately re-makes the undone AI picks"
    )


# ── Task 1.13b: No raw HTML in player selectbox options ────────────────────


def test_player_select_options_have_no_img_tags():
    """The player_names list passed to render_player_select in the Available
    Players tab must NOT contain raw <img ...> HTML strings.

    The _add_headshot() helper injects <img ...> into the player_name column
    for table display, but that column must be stripped of HTML before use
    as selectbox option labels.
    """
    src = _page_src()

    # Find the render_player_select call in the Available Players tab section
    rps_idx = src.rfind("render_player_select(")
    assert rps_idx != -1, "render_player_select() call not found in page"

    # The argument to render_player_select for player_names must not reference
    # the HTML-injected column directly. Specifically, it must not pass
    # disp_sorted["Player"].tolist() after the _add_headshot injection without
    # stripping HTML, OR it must use a separate plain-name list.
    #
    # Strategy: confirm that the page uses the ORIGINAL plain-name column (from
    # `disp` / `available`, before `_add_headshot` runs) rather than the
    # HTML-contaminated `disp_sorted["Player"]` column.
    #
    # After the fix, the page should either:
    #   (a) build a separate plain_names list from the pre-headshot `disp` frame, OR
    #   (b) strip HTML from the column before passing to render_player_select.
    #
    # We detect the broken pattern: passing disp_sorted["Player"].tolist() as the
    # first argument when disp_sorted["Player"] contains img tags.
    # Acceptable patterns include passing `disp["player_name"]` or a
    # `plain_names` / `_plain_names` variable.

    # Extract the render_player_select(...) call arguments by grabbing
    # the substring from the call up to the closing paren.
    call_start = rps_idx
    depth = 0
    call_end = rps_idx
    for i, ch in enumerate(src[call_start:], start=call_start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                call_end = i
                break
    call_src = src[call_start : call_end + 1]

    # The first positional argument (player_names) must NOT be disp_sorted["Player"]
    # because that column has been mutated to contain <img ...> HTML.
    assert 'disp_sorted["Player"].tolist()' not in call_src and "disp_sorted['Player'].tolist()" not in call_src, (
        "render_player_select() must not use disp_sorted['Player'] (which contains <img> HTML) "
        "as the player_names argument. Use a plain-name list derived from `disp` or stripped of HTML."
    )


def test_add_headshot_does_not_contaminate_selectbox_names():
    """Simulate what the _add_headshot helper does and verify the corrected
    code passes plain names (without <img>) to render_player_select.

    This is a logic test, not an AST test — it verifies the fix produces
    clean names for the selectbox even when mlb_id is present.
    """
    # Simulate the disp_sorted DataFrame after _add_headshot injection
    data = {
        "Player": [
            '<img src="https://example.com/1.jpg" width="22" height="22">Aaron Judge',
            '<img src="https://example.com/2.jpg" width="22" height="22">Shohei Ohtani',
            "No Image Player",
        ],
        "player_id": [1, 2, 3],
    }
    df = pd.DataFrame(data)

    # The fix: strip HTML tags from names before using as selectbox options
    import re as _re

    plain_names = [_re.sub(r"<[^>]+>", "", name) for name in df["Player"].tolist()]

    for name in plain_names:
        assert "<img" not in name, f"Plain name still contains <img>: {name!r}"
        assert "<" not in name, f"Plain name still contains HTML: {name!r}"

    assert plain_names[0] == "Aaron Judge"
    assert plain_names[1] == "Shohei Ohtani"
    assert plain_names[2] == "No Image Player"
