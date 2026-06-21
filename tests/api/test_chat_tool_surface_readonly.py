"""Guard: Bubba is READ + ANALYZE only — it must NEVER register a league-write
tool (owner product boundary, 2026-06-20). The user executes every transaction
manually in their provider's app; the AI only recommends. This locks the tool
surface so a future write/action tool can't silently slip in.
"""

from src.ai.tools import tool_specs

_BANNED = (
    "set_lineup",
    "add_drop",
    "add_player",
    "drop_player",
    "make_trade",
    "propose_trade",
    "execute",
    "submit",
    "place_",
    "transaction",
    "write",
    "mutate",
)


def test_no_league_write_tool_is_registered():
    names = [
        s.get("function", {}).get("name", "").lower()
        for s in tool_specs(web_search_enabled=True, deep_research_enabled=True)
    ]
    offenders = [n for n in names if any(b in n for b in _BANNED)]
    assert not offenders, f"Bubba must stay read-only; banned tool(s) registered: {offenders}"
