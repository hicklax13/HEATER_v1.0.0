"""Marcel projection fallback — populate the ``projections`` table from local
MLB ``season_stats`` when FanGraphs is unavailable.

On Railway the FanGraphs projection fetch returns HTTP 403 (Cloudflare blocks
the datacenter IP), so the ``projections`` table never populates and the player
pool ends up with no projection stats — free-agent "value" then renders 0 for
every player. Marcel (``src/marcel.py``) is pure compute over the MLB
``season_stats`` table (which IS available on the server) so it can fill the
projections blend with no network access.

This module is the DB seam: it reads ``players`` + ``season_stats`` via the
sanctioned ``get_connection()`` and writes ``system='marcel'`` rows shaped like
``create_blended_projections``'s output so the existing blender picks them up
(when Marcel is the only system present, ``blended`` == Marcel).
"""

from __future__ import annotations

import logging

from src.database import get_connection
from src.marcel import project_batch_marcel

logger = logging.getLogger(__name__)

# Per-player history horizon (Marcel uses up to 3 most-recent seasons).
_MARCEL_HISTORY_SEASONS = 3

# No usable per-player age column on the ``players`` table (only an optional
# ``birth_date`` that is frequently NULL), so default to Marcel's own neutral
# age. 28 is also ``project_batch_marcel``'s default.
_DEFAULT_AGE = 28

# Typical AB/PA ratio — used to derive an AB volume from Marcel's projected PA
# so the blender's avg = h/ab recomputation reproduces Marcel's AVG instead of
# zeroing it (Marcel emits rates + PA, not AB/H components).
_AB_PER_PA = 0.9

# Season-stat columns pulled per player type, mapped into the Marcel ``history``
# dict using the EXACT keys Marcel reads (see project_player_marcel).
_HITTER_HISTORY_COLS = ("r", "hr", "rbi", "sb", "avg", "obp", "pa")
_PITCHER_HISTORY_COLS = ("w", "l", "sv", "k", "era", "whip", "ip")


def _hitter_components(proj: dict[str, float]) -> dict[str, float]:
    """Derive AB/H/BB/HBP/SF from a Marcel hitter projection.

    The blender recomputes AVG = H/AB and OBP = (H+BB+HBP)/(AB+BB+HBP+SF) from
    components. Marcel only emits rates (avg, obp) + PA, so we back out a
    consistent component line that reproduces those rates (otherwise the blender
    would zero the rate stats).
    """
    pa = max(0.0, float(proj.get("pa", 0.0)))
    avg = max(0.0, float(proj.get("avg", 0.0)))
    obp = max(0.0, min(0.999, float(proj.get("obp", 0.0))))

    ab = round(pa * _AB_PER_PA)
    h = round(avg * ab)

    # OBP = (H + onbase_extras) / (AB + onbase_extras)  with SF folded in below.
    # Solve onbase_extras (bb+hbp) so the blender reproduces marcel obp:
    #   obp = (h + e) / (ab + e)  ->  e = (obp*ab - h) / (1 - obp)
    onbase_extras = 0
    if obp > 0.0 and obp < 1.0 and ab > 0:
        raw = (obp * ab - h) / (1.0 - obp)
        onbase_extras = max(0, round(raw))

    return {
        "pa": pa,
        "ab": float(ab),
        "h": float(h),
        "bb": float(onbase_extras),
        "hbp": 0.0,
        "sf": 0.0,
    }


def _pitcher_components(proj: dict[str, float]) -> dict[str, float]:
    """Derive ER/BB_allowed/H_allowed from a Marcel pitcher projection.

    The blender recomputes ERA = ER*9/IP and WHIP = (BB_allowed+H_allowed)/IP.
    Marcel only emits rates (era, whip) + IP, so we back out a component line
    that reproduces those rates (otherwise the blender would zero ERA/WHIP).
    """
    ip = max(0.0, float(proj.get("ip", 0.0)))
    era = max(0.0, float(proj.get("era", 0.0)))
    whip = max(0.0, float(proj.get("whip", 0.0)))

    er = round(era * ip / 9.0) if ip > 0 else 0
    baserunners = round(whip * ip) if ip > 0 else 0  # split is irrelevant to WHIP

    return {
        "ip": ip,
        "er": float(er),
        "bb_allowed": 0.0,
        "h_allowed": float(baserunners),
    }


def _load_players(conn) -> list[dict]:
    """Load all players with the fields Marcel needs."""
    cur = conn.execute("SELECT player_id, is_hitter FROM players")
    return [{"player_id": int(r[0]), "is_hitter": bool(r[1])} for r in cur.fetchall()]


def _load_history(conn, player_id: int, is_hitter: bool) -> list[dict]:
    """Load up to 3 most-recent seasons for a player as Marcel history dicts."""
    cols = _HITTER_HISTORY_COLS if is_hitter else _PITCHER_HISTORY_COLS
    select_cols = ", ".join(cols)
    cur = conn.execute(
        f"SELECT {select_cols} FROM season_stats "  # noqa: S608 - cols are a fixed allowlist
        "WHERE player_id = ? ORDER BY season DESC LIMIT ?",
        (player_id, _MARCEL_HISTORY_SEASONS),
    )
    history: list[dict] = []
    for row in cur.fetchall():
        history.append({col: (row[i] if row[i] is not None else 0.0) for i, col in enumerate(cols)})
    return history


def generate_marcel_projections(conn=None) -> int:
    """Compute Marcel projections from ``season_stats`` and write them as
    ``system='marcel'`` rows in the ``projections`` table.

    Pure compute (no FanGraphs / no network) — works on Railway where FanGraphs
    403s. Deletes any existing ``system='marcel'`` rows first, then inserts one
    row per player (idempotent). Never raises on a single malformed player
    (skips + logs a warning).

    Parameters
    ----------
    conn : sqlite3.Connection | None
        Optional open connection. When None, a connection is opened via
        ``get_connection()`` and closed before returning.

    Returns
    -------
    int
        The number of ``system='marcel'`` rows written.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()
    try:
        players = _load_players(conn)
        if not players:
            logger.info("Marcel fallback: no players in DB; nothing to project.")
            return 0

        # Build the Marcel batch input (each player carries its history).
        batch: list[dict] = []
        for p in players:
            try:
                hist = _load_history(conn, p["player_id"], p["is_hitter"])
            except Exception:
                logger.warning(
                    "Marcel fallback: failed to load history for player_id=%s; projecting from league mean.",
                    p["player_id"],
                    exc_info=True,
                )
                hist = []
            batch.append(
                {
                    "player_id": p["player_id"],
                    "is_hitter": p["is_hitter"],
                    "age": _DEFAULT_AGE,
                    "history": hist,
                }
            )

        projections = project_batch_marcel(batch)

        cur = conn.cursor()
        cur.execute("DELETE FROM projections WHERE system = 'marcel'")

        written = 0
        for player, proj in zip(batch, projections):
            try:
                row = _build_projection_row(player, proj)
            except Exception:
                logger.warning(
                    "Marcel fallback: failed to build projection row for player_id=%s; skipping.",
                    player.get("player_id"),
                    exc_info=True,
                )
                continue
            cur.execute(
                """
                INSERT INTO projections
                    (player_id, system, pa, ab, h, r, hr, rbi, sb, avg, obp,
                     bb, hbp, sf, ip, w, l, sv, k, era, whip, er, bb_allowed,
                     h_allowed, fip, xfip, siera)
                VALUES (?, 'marcel', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            written += 1

        conn.commit()
        logger.info("Marcel fallback: wrote %d projection rows.", written)
        return written
    finally:
        if owns_conn:
            conn.close()


def _build_projection_row(player: dict, proj: dict[str, float]) -> tuple:
    """Map a Marcel projection dict to the ``projections`` INSERT tuple.

    Counting stats pass through directly; rate stats (avg/obp/era/whip) are
    carried alongside derived components so the blender's rate recomputation
    reproduces the Marcel rates rather than zeroing them.
    """
    pid = player["player_id"]
    is_hitter = player["is_hitter"]

    if is_hitter:
        comp = _hitter_components(proj)
        return (
            pid,
            int(round(comp["pa"])),  # pa
            int(round(comp["ab"])),  # ab
            int(round(comp["h"])),  # h
            int(round(float(proj.get("r", 0.0)))),  # r
            int(round(float(proj.get("hr", 0.0)))),  # hr
            int(round(float(proj.get("rbi", 0.0)))),  # rbi
            int(round(float(proj.get("sb", 0.0)))),  # sb
            round(float(proj.get("avg", 0.0)), 3),  # avg
            round(float(proj.get("obp", 0.0)), 3),  # obp
            int(round(comp["bb"])),  # bb
            int(round(comp["hbp"])),  # hbp
            int(round(comp["sf"])),  # sf
            0.0,  # ip
            0,  # w
            0,  # l
            0,  # sv
            0,  # k
            0.0,  # era
            0.0,  # whip
            0,  # er
            0,  # bb_allowed
            0,  # h_allowed
            0.0,  # fip
            0.0,  # xfip
            0.0,  # siera
        )

    comp = _pitcher_components(proj)
    return (
        pid,
        0,  # pa
        0,  # ab
        0,  # h
        0,  # r
        0,  # hr
        0,  # rbi
        0,  # sb
        0.0,  # avg
        0.0,  # obp
        0,  # bb
        0,  # hbp
        0,  # sf
        round(comp["ip"], 1),  # ip
        int(round(float(proj.get("w", 0.0)))),  # w
        int(round(float(proj.get("l", 0.0)))),  # l
        int(round(float(proj.get("sv", 0.0)))),  # sv
        int(round(float(proj.get("k", 0.0)))),  # k
        round(float(proj.get("era", 0.0)), 2),  # era
        round(float(proj.get("whip", 0.0)), 2),  # whip
        int(round(comp["er"])),  # er
        int(round(comp["bb_allowed"])),  # bb_allowed
        int(round(comp["h_allowed"])),  # h_allowed
        0.0,  # fip
        0.0,  # xfip
        0.0,  # siera
    )
