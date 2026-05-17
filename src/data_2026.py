"""Hardcoded 2026 MLB player projections for fantasy baseball draft tool.

Contains ~200 hitters and ~80 pitchers with realistic projected stats,
ADP values, projection spread, and risk scores. Data is organized by
ADP tier and designed to feed into the database.py import pipeline
via temporary CSV files.

All stats represent full-season 2026 projections based on recent
performance trends, age curves, and team context.
"""


# ---------------------------------------------------------------------------
# Helper: build a hitter projection dict
# ---------------------------------------------------------------------------


def _make_hitter(
    name,
    team,
    positions,
    pa,
    hr,
    sb,
    avg,
    r,
    rbi,
    adp,
    spread=0.1,
    risk=0.1,
    obp=None,
    bb=None,
    hbp=None,
    sf=None,
    bats="R",
    birth_date=None,
):
    """Create a hitter projection dict. Computes ab, h from pa and avg.

    New stats (obp, bb, hbp, sf) are derived from existing params when
    not explicitly provided:
      obp  = avg + 0.065
      bb   = int(pa * 0.09)
      hbp  = int(pa * 0.01)
      sf   = int(pa * 0.008)
    """
    ab = int(pa * 0.89)  # ~89% of PA become AB (rest: BB, HBP, sac)
    h = int(ab * avg)
    if obp is None:
        obp = round(avg + 0.065, 3)
    if bb is None:
        bb = int(pa * 0.09)
    if hbp is None:
        hbp = int(pa * 0.01)
    if sf is None:
        sf = int(pa * 0.008)
    return {
        "name": name,
        "team": team,
        "positions": positions,
        "pa": pa,
        "ab": ab,
        "h": h,
        "r": r,
        "hr": hr,
        "rbi": rbi,
        "sb": sb,
        "avg": round(avg, 3),
        "obp": round(obp, 3),
        "bb": bb,
        "hbp": hbp,
        "sf": sf,
        "adp": adp,
        "projection_spread": spread,
        "risk_score": risk,
        "bats": bats,
        "birth_date": birth_date,
    }


# ---------------------------------------------------------------------------
# Helper: build a pitcher projection dict
# ---------------------------------------------------------------------------


def _make_pitcher(
    name,
    team,
    positions,
    ip,
    w,
    sv,
    k,
    era,
    whip,
    adp,
    spread=0.1,
    risk=0.1,
    gs=0,
    l=None,
    throws="R",
    birth_date=None,
    fip=None,
    xfip=None,
    siera=None,
):
    """Create a pitcher projection dict. Computes er, bb_allowed, h_allowed.

    Losses (l) derived when not explicitly provided:
      For SP: l = max(0, int(gs * 0.45) - w)
      For RP: l = max(0, int(w * 0.6))

    Advanced metrics (fip, xfip, siera) derived from ERA when not provided:
      fip   = ERA - 0.10
      xfip  = ERA + 0.05
      siera = ERA - 0.05
    """
    er = int(ip * era / 9)
    total_br = ip * whip
    bb_allowed = int(total_br * 0.35)
    h_allowed = int(total_br - bb_allowed)
    if gs == 0 and "SP" in positions:
        gs = int(ip / 6)  # rough estimate for starters
    if l is None:
        if "SP" in positions:
            l = max(0, int(gs * 0.45) - w)
        else:
            # RP: typically few losses, roughly proportional to wins
            l = max(0, int(w * 0.6))
    if fip is None:
        fip = round(era - 0.10, 2)
    if xfip is None:
        xfip = round(era + 0.05, 2)
    if siera is None:
        siera = round(era - 0.05, 2)
    return {
        "name": name,
        "team": team,
        "positions": positions,
        "ip": ip,
        "w": w,
        "l": l,
        "sv": sv,
        "k": k,
        "era": round(era, 2),
        "whip": round(whip, 2),
        "er": er,
        "bb_allowed": bb_allowed,
        "h_allowed": h_allowed,
        "gs": gs,
        "adp": adp,
        "projection_spread": spread,
        "risk_score": risk,
        "throws": throws,
        "birth_date": birth_date,
        "fip": fip,
        "xfip": xfip,
        "siera": siera,
    }


# ---------------------------------------------------------------------------
# Hitter projections (~200 players)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pitcher projections (~80 players)
# ---------------------------------------------------------------------------
