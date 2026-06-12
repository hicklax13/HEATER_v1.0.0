"""AI tool surface over the existing service layer + the guarded SQL + refresh queue.

Tool schemas are OpenAI function-calling format (LiteLLM's lingua franca). All
dispatch returns a JSON string (the tool result the model reads). Read tools are
side-effect-free; request_refresh is the only (queued) write. web_search and
deep_research are only offered when the user enables them in the chat window.
"""

from __future__ import annotations

import json


def _fn(name: str, description: str, properties: dict | None = None, required: list | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                **({"required": required} if required else {}),
            },
        },
    }


def tool_specs(web_search_enabled: bool = False, deep_research_enabled: bool = False) -> list[dict]:
    specs = [
        _fn(
            "query_data",
            "Run a READ-ONLY SQL SELECT against the HEATER database to answer questions about "
            "ANY data (players, stats, projections, standings, rosters, trades, news, etc.). "
            "Use the schema in the system prompt. SELECT only.",
            {"sql": {"type": "string", "description": "A single SELECT statement."}},
            ["sql"],
        ),
        _fn(
            "get_player",
            "Look up a player's full row (projections, YTD, statcast) by name.",
            {"name": {"type": "string"}},
            ["name"],
        ),
        _fn(
            "compare_players",
            "Compare two players side by side (full rows) by name.",
            {"name_a": {"type": "string"}, "name_b": {"type": "string"}},
            ["name_a", "name_b"],
        ),
        _fn("get_my_team", "The viewer's own roster (their team's players)."),
        _fn("get_standings", "Current league standings (all 12 teams, category ranks)."),
        _fn(
            "get_free_agents",
            "Available free agents (top by ownership/value).",
            {"limit": {"type": "integer", "description": "Max FAs to return (default 50)."}},
        ),
        _fn(
            "request_refresh",
            "Request a data refresh. 'source' is optional/advisory (any request triggers a full "
            "refresh); omit to refresh everything. Returns a request_id; the scheduler runs it. Use sparingly.",
            {"source": {"type": "string"}},
        ),
    ]
    if web_search_enabled:
        specs.append(
            _fn(
                "web_search",
                "Search the public web (DuckDuckGo) for current information not in the HEATER "
                "database (injury news, recent performance, articles). Returns titles, URLs, snippets.",
                {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "Default 5."},
                },
                ["query"],
            )
        )
    if deep_research_enabled:
        specs.append(
            _fn(
                "deep_research",
                "Deeper web research: searches, fetches the top pages, and returns condensed sources "
                "(title/url/content) for you to synthesize into a cited answer. Use for open-ended "
                "research questions. Cite the URLs you use.",
                {"query": {"type": "string"}},
                ["query"],
            )
        )
    return specs


def dispatch_tool(name: str, args: dict, user_id: int) -> str:
    try:
        if name == "query_data":
            from src.ai.sql_tool import run_read_only_sql

            return json.dumps(run_read_only_sql(args.get("sql", "")), default=str)
        if name == "get_player":
            player_name = str(args.get("name", "")).strip()
            if not player_name:
                return json.dumps({"error": "Missing required arg: name"})
            return _get_player(player_name)
        if name == "compare_players":
            return _compare_players(str(args.get("name_a", "")), str(args.get("name_b", "")))
        if name == "get_my_team":
            return _get_my_team()
        if name == "get_standings":
            return _get_standings()
        if name == "get_free_agents":
            return _get_free_agents(int(args.get("limit", 50) or 50))
        if name == "request_refresh":
            from src.ai.refresh_queue import request_refresh

            rid = request_refresh(args.get("source", "all"), requested_by=user_id)
            return json.dumps({"request_id": rid, "status": "pending"})
        if name == "web_search":
            from src.ai.search import web_search

            q = str(args.get("query", "")).strip()
            if not q:
                return json.dumps({"error": "Missing required arg: query"})
            return json.dumps(web_search(q, max_results=int(args.get("max_results", 5) or 5)), default=str)
        if name == "deep_research":
            from src.ai.search import deep_research

            q = str(args.get("query", "")).strip()
            if not q:
                return json.dumps({"error": "Missing required arg: query"})
            return json.dumps(deep_research(q), default=str)
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as exc:  # never crash the agent loop
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _player_rows(names: list[str], limit_each: int = 1):
    from src.database import load_player_pool

    pool = load_player_pool()
    col = "player_name" if "player_name" in pool.columns else "name"
    out = []
    for name in names:
        n = name.strip().lower()
        if not n:
            continue
        hit = pool[pool[col].astype(str).str.lower() == n]
        if hit.empty:
            hit = pool[pool[col].astype(str).str.lower().str.contains(n, na=False)]
        out.append(hit.head(limit_each))
    return out


def _get_player(name: str) -> str:
    rows = _player_rows([name], limit_each=3)
    return rows[0].to_json(orient="records") if rows else json.dumps([])


def _compare_players(name_a: str, name_b: str) -> str:
    if not name_a.strip() or not name_b.strip():
        return json.dumps({"error": "Both name_a and name_b are required."})
    import pandas as pd

    rows = _player_rows([name_a, name_b], limit_each=1)
    combined = pd.concat([r for r in rows if not r.empty]) if rows else pd.DataFrame()
    return combined.to_json(orient="records")


def _get_my_team() -> str:
    from src.auth import resolve_viewer_team_name
    from src.yahoo_data_service import get_yahoo_data_service

    rosters = get_yahoo_data_service().get_rosters()
    if rosters is None or rosters.empty:
        return json.dumps({"error": "no roster data"})
    team = resolve_viewer_team_name(rosters)
    if not team or "team_name" not in rosters.columns:
        return json.dumps({"error": "could not resolve your team"})
    mine = rosters[rosters["team_name"].astype(str) == str(team)]
    return mine.to_json(orient="records")


def _get_standings() -> str:
    from src.yahoo_data_service import get_yahoo_data_service

    df = get_yahoo_data_service().get_standings()
    return df.to_json(orient="records") if df is not None else json.dumps({"error": "no standings"})


def _get_free_agents(limit: int) -> str:
    from src.yahoo_data_service import get_yahoo_data_service

    df = get_yahoo_data_service().get_free_agents(max_players=max(1, min(limit, 500)))
    if df is None or df.empty:
        return json.dumps({"error": "no free agents"})
    return df.head(limit).to_json(orient="records")
