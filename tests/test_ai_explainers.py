"""explain_metric assembles the formula + registry weights + provided inputs.
DB-free: the registry is pure; inputs are passed as params, not fetched."""

import json

from src.ai.tools import dispatch_tool, tool_specs


def test_explain_metric_in_specs():
    names = {t["function"]["name"] for t in tool_specs()}
    assert "explain_metric" in names


def test_explain_metric_stream_score_weights_match_registry():
    from src.optimizer.constants_registry import CONSTANTS_REGISTRY

    out = json.loads(dispatch_tool("explain_metric", {"kind": "stream_score"}, user_id=99))
    assert out["kind"] == "stream_score"
    assert out["formula"]  # a non-empty formula string
    by_key = {c["key"]: c for c in out["components"]}
    assert set(by_key) == {"matchup", "sgp", "form", "lineup", "env", "winprob"}
    for key, comp in by_key.items():
        assert comp["weight"] == CONSTANTS_REGISTRY[f"stream_score_w_{key}"].value


def test_explain_metric_surfaces_provided_component_values():
    out = json.loads(
        dispatch_tool(
            "explain_metric",
            {"kind": "stream_score", "params": {"components": {"matchup": 0.4, "sgp": -0.1}}},
            user_id=99,
        )
    )
    by_key = {c["key"]: c for c in out["components"]}
    assert by_key["matchup"]["value"] == 0.4
    assert by_key["sgp"]["value"] == -0.1
    # a component not supplied stays None (recipe, not a fabricated value)
    assert by_key["form"]["value"] is None


def test_explain_metric_dcv_returns_formula_and_weights():
    out = json.loads(dispatch_tool("explain_metric", {"kind": "dcv"}, user_id=99))
    assert out["kind"] == "dcv" and out["formula"]
    assert isinstance(out["components"], list)


def test_explain_metric_trade_grade_supported():
    out = json.loads(dispatch_tool("explain_metric", {"kind": "trade_grade"}, user_id=99))
    assert out["kind"] == "trade_grade" and out["formula"]


def test_explain_metric_trade_grade_not_over_inclusive():
    """The grade derives from SGP marginals, NOT from the broad league_avg_*
    baselines that a loose 'engine' substring used to drag in (they don't feed
    the grade). Components must be bounded + must not present those baselines as
    grade weights."""
    out = json.loads(dispatch_tool("explain_metric", {"kind": "trade_grade"}, user_id=99))
    keys = {c["key"] for c in out["components"]}
    assert not ({"league_avg_era", "league_avg_whip", "league_avg_woba"} & keys)
    # the old loose filter surfaced 3 unrelated baselines; the anchored filter is
    # bounded (formula carries the recipe; few/no registered weights compose it)
    assert len(out["components"]) <= 2


def test_explain_metric_non_stream_note_is_honest_not_exact_formula():
    """For dcv/trade_grade/start_score the components are area-related registered
    constants, NOT a guaranteed exact formula — the note must say so (avoid a
    confidently-wrong recipe claim). stream_score stays the exact 6-weight blend."""
    for kind in ("dcv", "trade_grade", "start_score"):
        out = json.loads(dispatch_tool("explain_metric", {"kind": kind}, user_id=99))
        note = out["note"].lower()
        assert "not" in note and ("exhaustive" in note or "exact" in note)


def test_explain_metric_start_score_supported():
    out = json.loads(dispatch_tool("explain_metric", {"kind": "start_score"}, user_id=99))
    assert out["kind"] == "start_score" and out["formula"]
    keys = {c["key"] for c in out["components"]}
    # start_sit.py constants only — no bleed from other modules
    assert keys == {"home_advantage", "away_discount"}


def test_explain_metric_unknown_kind_is_graceful():
    out = json.loads(dispatch_tool("explain_metric", {"kind": "bogus"}, user_id=99))
    assert "error" in out and "bogus" in out["error"]


def test_explain_metric_missing_kind_is_graceful():
    out = json.loads(dispatch_tool("explain_metric", {}, user_id=99))
    assert "error" in out


def test_new_explainer_tools_are_additive_to_specs():
    """The three explainer tools register WITHOUT removing any existing tool —
    the live Streamlit chat surface only grows."""
    from src.ai.tools import tool_specs

    names = {t["function"]["name"] for t in tool_specs()}
    # original surface still present
    assert {
        "query_data",
        "get_player",
        "compare_players",
        "get_my_team",
        "get_standings",
        "get_free_agents",
        "request_refresh",
    }.issubset(names)
    # new explainers added
    assert {"explain_constant", "list_constants", "explain_metric"}.issubset(names)


def test_chat_drain_unchanged_with_explainers_registered(monkeypatch):
    """FROZEN REFERENCE (additive): with the explainer tools in the registry,
    src/ai/providers.chat() still drains to the byte-identical pre-WS5 dict. The
    live Streamlit app (src/ai/chat.py) is provably unchanged."""
    from types import SimpleNamespace

    from src.ai import providers

    def _delta(content=None):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))])

    def _rebuilt(content=None, tool_calls=None, in_tok=10, out_tok=5):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
            usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
        )

    monkeypatch.setattr(providers, "_completion", lambda **kw: iter([_delta("Hi "), _delta("there")]))
    monkeypatch.setattr(providers, "_rebuild", lambda chunks, messages: _rebuilt(content="Hi there"))
    out = providers.chat(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        user_id=99,
    )
    assert out == {"content": "Hi there", "tokens_in": 10, "tokens_out": 5, "tool_trace": []}
