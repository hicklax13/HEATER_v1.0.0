"""AI tool surface over the existing service layer + the guarded SQL + refresh queue.

Tool schemas are OpenAI function-calling format (LiteLLM's lingua franca). All
dispatch returns a JSON string (the tool result the model reads). Read tools are
side-effect-free; request_refresh is the only (queued) write.
"""

from __future__ import annotations

import json


def tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "query_data",
                "description": (
                    "Run a READ-ONLY SQL SELECT against the HEATER database to answer "
                    "questions about any data (players, stats, projections, standings, "
                    "rosters, trades, etc.). Use the schema in the system prompt. SELECT only."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
                    "required": ["sql"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_player",
                "description": "Look up a player's full row (projections, YTD, statcast) by name.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_standings",
                "description": "Current league standings (all 12 teams, category ranks).",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "request_refresh",
                "description": (
                    "Request a data refresh for a source (e.g. 'players', 'yahoo', or 'all'). "
                    "Returns a request_id; the background scheduler runs it. Use sparingly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                },
            },
        },
    ]


def dispatch_tool(name: str, args: dict, user_id: int) -> str:
    try:
        if name == "query_data":
            from src.ai.sql_tool import run_read_only_sql

            return json.dumps(run_read_only_sql(args.get("sql", "")), default=str)
        if name == "get_player":
            return _get_player(args.get("name", ""))
        if name == "get_standings":
            return _get_standings()
        if name == "request_refresh":
            from src.ai.refresh_queue import request_refresh

            rid = request_refresh(args.get("source", "all"), requested_by=user_id)
            return json.dumps({"request_id": rid, "status": "pending"})
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as exc:  # never crash the agent loop
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _get_player(name: str) -> str:
    try:
        from src.database import load_player_pool

        pool = load_player_pool()
        col = "player_name" if "player_name" in pool.columns else "name"
        hit = pool[pool[col].astype(str).str.lower() == name.strip().lower()]
        if hit.empty:
            hit = pool[pool[col].astype(str).str.lower().str.contains(name.strip().lower(), na=False)]
        return hit.head(3).to_json(orient="records")
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _get_standings() -> str:
    try:
        from src.yahoo_data_service import get_yahoo_data_service

        df = get_yahoo_data_service().get_standings()
        return df.to_json(orient="records") if df is not None else json.dumps({"error": "no standings"})
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
