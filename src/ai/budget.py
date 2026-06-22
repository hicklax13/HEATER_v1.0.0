"""Per-user daily AI spend cap + usage ledger.

The cap applies to the admin SHARED key (the operator pays). A user on their own
key is exempt by default. Caps are admin-set in app_settings; sensible default.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.app_settings import get_setting, set_setting

_DAILY_CAP_SETTING = "ai_daily_cap_usd"
_DEFAULT_DAILY_CAP_USD = 1.00


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def daily_cap_usd() -> float:
    raw = get_setting(_DAILY_CAP_SETTING)
    if raw is None:
        return _DEFAULT_DAILY_CAP_USD
    try:
        return float(raw)
    except (ValueError, TypeError):
        return _DEFAULT_DAILY_CAP_USD


def set_daily_cap(usd: float, admin_id: int) -> None:
    # Clamp to >= 0: a negative cap would make is_over_cap True for everyone
    # (spent >= negative), silently locking all users out of chat.
    set_setting(_DAILY_CAP_SETTING, str(max(0.0, float(usd))), admin_id)


def record_usage(user_id: int, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO ai_usage_ledger (user_id, day, tokens_in, tokens_out, cost_usd)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, day) DO UPDATE SET
                tokens_in = tokens_in + excluded.tokens_in,
                tokens_out = tokens_out + excluded.tokens_out,
                cost_usd = cost_usd + excluded.cost_usd
            """,
            (user_id, _today(), int(tokens_in), int(tokens_out), max(0.0, float(cost_usd))),
        )
        conn.commit()
    finally:
        conn.close()


def spent_today(user_id: int) -> float:
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT cost_usd FROM ai_usage_ledger WHERE user_id = ? AND day = ?",
            (user_id, _today()),
        ).fetchone()
        return float(row["cost_usd"]) if row is not None else 0.0
    finally:
        conn.close()


def is_over_cap(user_id: int, on_own_key: bool = False, cap_usd: float | None = None) -> bool:
    if on_own_key:
        return False
    cap = cap_usd if cap_usd is not None else daily_cap_usd()
    return spent_today(user_id) >= cap
