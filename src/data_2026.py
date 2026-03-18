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
    name, team, positions, pa, hr, sb, avg, r, rbi, adp,
    spread=0.1, risk=0.1, obp=None, bb=None, hbp=None, sf=None,
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
    }


# ---------------------------------------------------------------------------
# Helper: build a pitcher projection dict
# ---------------------------------------------------------------------------


def _make_pitcher(name, team, positions, ip, w, sv, k, era, whip, adp, spread=0.1, risk=0.1, gs=0, l=None):
    """Create a pitcher projection dict. Computes er, bb_allowed, h_allowed.

    Losses (l) derived when not explicitly provided:
      For SP: l = max(0, int(gs * 0.45) - w)
      For RP: l = max(0, int(w * 0.6))
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
    }


# ---------------------------------------------------------------------------
# Hitter projections (~200 players)
# ---------------------------------------------------------------------------


def get_hitter_projections() -> list[dict]:
    """Return ~200 hitter projections for 2026 drafts."""
    return [
        # ==================================================================
        # ELITE TIER  (ADP 1-25)
        # ==================================================================
        _make_hitter("Shohei Ohtani", "LAD", "DH", 600, 45, 20, 0.285, 105, 110, 1, 0.08, 0.15),
        _make_hitter("Bobby Witt Jr.", "KC", "SS", 670, 32, 40, 0.290, 110, 100, 2, 0.07, 0.08),
        _make_hitter("Aaron Judge", "NYY", "OF", 620, 48, 5, 0.280, 108, 120, 3, 0.08, 0.18),
        _make_hitter("Juan Soto", "NYM", "OF", 660, 35, 5, 0.290, 112, 105, 4, 0.06, 0.08),
        _make_hitter("Elly De La Cruz", "CIN", "SS", 620, 28, 62, 0.260, 105, 85, 5, 0.30, 0.12),
        _make_hitter("Gunnar Henderson", "BAL", "SS,3B", 650, 36, 18, 0.275, 108, 95, 6, 0.12, 0.08),
        _make_hitter("Ronald Acuna Jr.", "ATL", "OF", 580, 28, 45, 0.280, 100, 85, 7, 0.25, 0.35),
        _make_hitter("Fernando Tatis Jr.", "SD", "SS,OF", 580, 34, 28, 0.275, 100, 92, 8, 0.18, 0.28),
        _make_hitter("Mookie Betts", "LAD", "SS,OF", 620, 28, 15, 0.280, 105, 88, 9, 0.08, 0.15),
        _make_hitter("Yordan Alvarez", "HOU", "OF,DH", 590, 38, 3, 0.285, 95, 108, 10, 0.08, 0.18),
        _make_hitter("Freddie Freeman", "LAD", "1B", 640, 25, 8, 0.295, 100, 95, 11, 0.06, 0.10),
        _make_hitter("Corey Seager", "TEX", "SS", 600, 33, 4, 0.280, 98, 100, 12, 0.08, 0.10),
        _make_hitter("Kyle Tucker", "CHC", "OF", 610, 30, 22, 0.280, 100, 95, 13, 0.10, 0.15),
        _make_hitter("Trea Turner", "PHI", "SS", 600, 22, 30, 0.280, 100, 80, 14, 0.10, 0.18),
        _make_hitter("Jose Ramirez", "CLE", "3B", 620, 30, 20, 0.275, 95, 105, 16, 0.07, 0.10),
        _make_hitter("Julio Rodriguez", "SEA", "OF", 620, 28, 30, 0.270, 98, 88, 17, 0.18, 0.12),
        _make_hitter("Rafael Devers", "BOS", "3B", 630, 32, 4, 0.280, 95, 105, 19, 0.08, 0.10),
        _make_hitter("Corbin Carroll", "ARI", "OF", 620, 22, 35, 0.270, 100, 78, 20, 0.22, 0.12),
        _make_hitter("Matt Olson", "ATL", "1B", 620, 38, 3, 0.250, 95, 110, 22, 0.12, 0.08),
        _make_hitter("Marcus Semien", "TEX", "2B", 650, 25, 15, 0.270, 100, 85, 24, 0.08, 0.07),
        # ==================================================================
        # UPPER-MID TIER  (ADP 25-75)
        # ==================================================================
        _make_hitter("Jackson Chourio", "MIL", "OF", 600, 25, 25, 0.275, 92, 82, 26, 0.28, 0.10),
        _make_hitter("Adley Rutschman", "BAL", "C", 580, 24, 5, 0.275, 85, 90, 28, 0.10, 0.08),
        _make_hitter("Vladimir Guerrero Jr.", "TOR", "1B", 640, 32, 5, 0.285, 92, 100, 30, 0.08, 0.08),
        _make_hitter("Bryce Harper", "PHI", "1B,DH", 590, 30, 8, 0.275, 90, 95, 32, 0.10, 0.18),
        _make_hitter("Jarren Duran", "BOS", "OF", 610, 18, 35, 0.280, 100, 72, 34, 0.18, 0.10),
        _make_hitter("Francisco Lindor", "NYM", "SS", 620, 28, 18, 0.265, 95, 88, 36, 0.08, 0.10),
        _make_hitter("James Wood", "WSH", "OF", 560, 22, 20, 0.265, 85, 78, 38, 0.35, 0.15),
        _make_hitter("CJ Abrams", "WSH", "SS", 600, 20, 35, 0.265, 95, 72, 40, 0.22, 0.10),
        _make_hitter("Manny Machado", "SD", "3B", 600, 28, 10, 0.270, 88, 92, 42, 0.08, 0.10),
        _make_hitter("Marcell Ozuna", "ATL", "OF,DH", 580, 35, 3, 0.270, 85, 100, 44, 0.12, 0.12),
        _make_hitter("Pete Alonso", "NYM", "1B", 600, 35, 3, 0.250, 85, 100, 46, 0.12, 0.08),
        _make_hitter("Bo Bichette", "TOR", "SS", 560, 18, 15, 0.275, 80, 75, 48, 0.18, 0.25),
        _make_hitter("Austin Riley", "ATL", "3B", 580, 30, 3, 0.270, 82, 95, 50, 0.12, 0.15),
        _make_hitter("Ozzie Albies", "ATL", "2B", 580, 22, 18, 0.270, 88, 80, 52, 0.12, 0.18),
        _make_hitter("Alex Bregman", "HOU", "3B", 600, 22, 5, 0.270, 85, 85, 54, 0.08, 0.08),
        _make_hitter("Luis Robert Jr.", "CHW", "OF", 480, 28, 18, 0.270, 78, 80, 56, 0.25, 0.38),
        _make_hitter("Michael Harris II", "ATL", "OF", 580, 22, 22, 0.275, 88, 78, 58, 0.15, 0.12),
        _make_hitter("Anthony Santander", "TOR", "OF", 590, 35, 5, 0.255, 85, 98, 60, 0.12, 0.10),
        _make_hitter("Jazz Chisholm Jr.", "NYY", "3B,2B", 550, 24, 22, 0.260, 85, 78, 62, 0.22, 0.22),
        _make_hitter("Willy Adames", "SF", "SS", 580, 28, 10, 0.255, 88, 90, 64, 0.10, 0.10),
        _make_hitter("George Springer", "TOR", "OF", 550, 22, 10, 0.265, 82, 75, 66, 0.10, 0.18),
        _make_hitter("Ketel Marte", "ARI", "2B,DH", 560, 25, 8, 0.280, 85, 82, 68, 0.10, 0.15),
        _make_hitter("Seiya Suzuki", "CHC", "OF", 540, 25, 10, 0.275, 80, 85, 70, 0.12, 0.12),
        _make_hitter("Colton Cowser", "BAL", "OF", 570, 24, 12, 0.260, 85, 78, 72, 0.22, 0.10),
        _make_hitter("Riley Greene", "DET", "OF", 590, 22, 12, 0.270, 88, 80, 74, 0.15, 0.10),
        # ==================================================================
        # MID TIER  (ADP 75-150)
        # ==================================================================
        _make_hitter("Will Smith", "LAD", "C", 530, 22, 3, 0.265, 75, 82, 76, 0.10, 0.08),
        _make_hitter("Dansby Swanson", "CHC", "SS", 580, 22, 12, 0.255, 82, 78, 78, 0.10, 0.12),
        _make_hitter("Bryan Reynolds", "PIT", "OF", 580, 22, 10, 0.270, 80, 80, 80, 0.10, 0.08),
        _make_hitter("Salvador Perez", "KC", "C,DH", 560, 25, 2, 0.260, 70, 90, 82, 0.08, 0.12),
        _make_hitter("Christian Yelich", "MIL", "OF", 520, 18, 15, 0.270, 80, 70, 84, 0.12, 0.25),
        _make_hitter("Tommy Edman", "LAD", "2B,SS,OF", 550, 12, 28, 0.265, 82, 58, 86, 0.15, 0.18),
        _make_hitter("Giancarlo Stanton", "NYY", "OF,DH", 480, 30, 2, 0.245, 72, 85, 88, 0.15, 0.35),
        _make_hitter("Randy Arozarena", "SEA", "OF", 550, 20, 18, 0.260, 78, 72, 90, 0.15, 0.12),
        _make_hitter("Nolan Arenado", "HOU", "3B", 550, 22, 3, 0.265, 72, 82, 92, 0.08, 0.12),
        _make_hitter("Ezequiel Tovar", "COL", "SS", 580, 20, 15, 0.265, 78, 72, 94, 0.22, 0.10),
        _make_hitter("Brandon Nimmo", "NYM", "OF", 560, 20, 5, 0.268, 82, 75, 96, 0.10, 0.10),
        _make_hitter("Cedric Mullins", "BAL", "OF", 540, 15, 25, 0.258, 78, 62, 98, 0.15, 0.12),
        _make_hitter("Cody Bellinger", "CHC", "1B,OF", 540, 22, 10, 0.265, 78, 78, 100, 0.15, 0.15),
        _make_hitter("Mike Trout", "LAA", "OF", 400, 25, 5, 0.260, 65, 65, 102, 0.30, 0.50),
        _make_hitter("Yandy Diaz", "TB", "1B,3B", 580, 18, 3, 0.280, 75, 78, 104, 0.08, 0.08),
        _make_hitter("Ryan McMahon", "COL", "3B,2B", 560, 22, 5, 0.255, 78, 80, 106, 0.12, 0.10),
        _make_hitter("J.P. Crawford", "SEA", "SS", 560, 10, 8, 0.270, 72, 60, 108, 0.08, 0.08),
        _make_hitter("Lars Nootbaar", "STL", "OF", 500, 18, 8, 0.258, 72, 65, 110, 0.15, 0.15),
        _make_hitter("Isaac Paredes", "CHC", "3B,1B", 540, 20, 3, 0.258, 70, 75, 112, 0.12, 0.08),
        _make_hitter("Masataka Yoshida", "BOS", "OF,DH", 500, 15, 5, 0.285, 68, 72, 114, 0.10, 0.15),
        _make_hitter("Tyler O'Neill", "BOS", "OF", 450, 25, 8, 0.245, 68, 72, 116, 0.22, 0.30),
        _make_hitter("Teoscar Hernandez", "LAD", "OF", 560, 25, 5, 0.258, 78, 85, 118, 0.12, 0.10),
        _make_hitter("Gleyber Torres", "DET", "2B", 560, 18, 8, 0.262, 75, 70, 120, 0.12, 0.10),
        _make_hitter("Alec Bohm", "PHI", "3B,1B", 580, 18, 5, 0.270, 72, 82, 122, 0.10, 0.08),
        _make_hitter("Spencer Torkelson", "DET", "1B", 520, 22, 3, 0.248, 68, 75, 124, 0.25, 0.12),
        _make_hitter("Wyatt Langford", "TEX", "OF", 520, 18, 15, 0.258, 72, 68, 126, 0.30, 0.12),
        _make_hitter("Jordan Walker", "STL", "OF", 480, 20, 10, 0.255, 68, 72, 128, 0.32, 0.15),
        _make_hitter("Brice Turang", "MIL", "2B,SS", 550, 8, 30, 0.262, 75, 52, 130, 0.18, 0.10),
        _make_hitter("Vinnie Pasquantino", "KC", "1B,DH", 560, 22, 2, 0.268, 72, 82, 132, 0.12, 0.12),
        _make_hitter("Wilyer Abreu", "BOS", "OF", 540, 18, 12, 0.265, 75, 68, 134, 0.18, 0.10),
        _make_hitter("Eloy Jimenez", "BAL", "OF,DH", 450, 22, 2, 0.260, 60, 72, 136, 0.18, 0.30),
        _make_hitter("Josh Lowe", "TB", "OF", 520, 18, 22, 0.252, 75, 65, 138, 0.18, 0.12),
        _make_hitter("Zack Gelof", "OAK", "2B,3B", 540, 20, 18, 0.248, 75, 68, 140, 0.25, 0.12),
        _make_hitter("Jake Cronenworth", "SD", "1B,2B", 560, 18, 5, 0.262, 72, 72, 142, 0.10, 0.08),
        _make_hitter("Royce Lewis", "MIN", "SS,3B", 400, 22, 8, 0.268, 60, 62, 144, 0.30, 0.45),
        _make_hitter("Evan Carter", "TEX", "OF", 480, 16, 15, 0.258, 70, 60, 146, 0.30, 0.18),
        _make_hitter("Max Muncy", "LAD", "3B,1B", 530, 25, 3, 0.235, 78, 78, 148, 0.12, 0.15),
        # ==================================================================
        # LATE TIER  (ADP 150-250)
        # ==================================================================
        _make_hitter("J.D. Martinez", "NYM", "DH", 500, 22, 2, 0.260, 68, 78, 150, 0.10, 0.12),
        _make_hitter("Jose Abreu", "HOU", "1B,DH", 450, 18, 2, 0.255, 55, 68, 152, 0.12, 0.15),
        _make_hitter("Cal Raleigh", "SEA", "C", 510, 25, 2, 0.232, 65, 78, 154, 0.12, 0.08),
        _make_hitter("Wander Franco", "TB", "SS", 350, 12, 10, 0.270, 50, 42, 156, 0.50, 0.55),
        _make_hitter("Andrew Vaughn", "CHW", "1B,OF", 500, 18, 3, 0.258, 62, 70, 158, 0.15, 0.10),
        _make_hitter("Carlos Correa", "MIN", "SS", 500, 18, 4, 0.265, 68, 72, 161, 0.12, 0.22),
        _make_hitter("Alex Verdugo", "NYY", "OF", 520, 14, 5, 0.268, 68, 65, 164, 0.10, 0.08),
        _make_hitter("Ha-Seong Kim", "SD", "SS,2B,3B", 480, 12, 18, 0.258, 68, 55, 167, 0.15, 0.18),
        _make_hitter("Rhys Hoskins", "MIL", "1B", 500, 25, 2, 0.238, 70, 78, 170, 0.15, 0.22),
        _make_hitter("Andres Gimenez", "CLE", "2B", 540, 14, 15, 0.260, 72, 60, 173, 0.12, 0.12),
        _make_hitter("Gavin Lux", "LAD", "2B,SS", 480, 12, 8, 0.268, 65, 55, 176, 0.18, 0.18),
        _make_hitter("Christopher Morel", "CHC", "3B,OF", 480, 22, 12, 0.232, 68, 68, 179, 0.25, 0.12),
        _make_hitter("Noelvi Marte", "CIN", "3B,SS", 480, 18, 12, 0.262, 65, 68, 182, 0.30, 0.15),
        _make_hitter("Anthony Volpe", "NYY", "SS", 560, 15, 18, 0.248, 75, 62, 185, 0.15, 0.10),
        _make_hitter("Thairo Estrada", "SF", "2B,SS", 500, 12, 15, 0.262, 65, 55, 188, 0.12, 0.10),
        _make_hitter("Justin Turner", "TOR", "3B,DH", 450, 15, 2, 0.265, 55, 65, 191, 0.10, 0.18),
        _make_hitter("Brandon Lowe", "TB", "2B,1B", 450, 18, 5, 0.248, 62, 65, 194, 0.18, 0.22),
        _make_hitter("Brendan Donovan", "STL", "2B,OF,1B", 520, 12, 8, 0.268, 68, 60, 197, 0.12, 0.08),
        _make_hitter("William Contreras", "MIL", "C,DH", 500, 18, 3, 0.260, 62, 72, 199, 0.12, 0.10),
        _make_hitter("Lane Thomas", "CLE", "OF", 500, 15, 18, 0.252, 68, 58, 201, 0.15, 0.12),
        _make_hitter("Hunter Renfroe", "KC", "OF", 450, 22, 3, 0.240, 58, 68, 203, 0.15, 0.12),
        _make_hitter("Xander Bogaerts", "SD", "SS,DH", 420, 12, 3, 0.268, 55, 58, 205, 0.15, 0.30),
        _make_hitter("Keibert Ruiz", "WSH", "C", 480, 12, 2, 0.258, 50, 55, 207, 0.12, 0.10),
        _make_hitter("Jurickson Profar", "SD", "OF,1B", 540, 18, 8, 0.258, 72, 70, 209, 0.12, 0.10),
        _make_hitter("Josh Naylor", "CLE", "1B,DH", 530, 22, 3, 0.252, 68, 82, 211, 0.12, 0.10),
        _make_hitter("Ceddanne Rafaela", "BOS", "SS,OF", 500, 14, 18, 0.255, 68, 55, 213, 0.28, 0.12),
        _make_hitter("Joey Meneses", "WSH", "1B", 450, 15, 2, 0.262, 52, 62, 215, 0.12, 0.10),
        _make_hitter("Trevor Story", "BOS", "SS", 380, 15, 10, 0.248, 52, 50, 217, 0.25, 0.42),
        _make_hitter("Christian Walker", "HOU", "1B", 540, 25, 3, 0.248, 72, 80, 219, 0.10, 0.10),
        _make_hitter("J.T. Realmuto", "PHI", "C", 450, 15, 8, 0.258, 58, 58, 221, 0.12, 0.22),
        _make_hitter("Nelson Cruz", "CLE", "DH", 380, 18, 1, 0.245, 48, 58, 223, 0.12, 0.18),
        _make_hitter("Austin Hays", "PHI", "OF", 420, 14, 8, 0.262, 55, 55, 226, 0.18, 0.22),
        _make_hitter("Kerry Carpenter", "DET", "OF,DH", 450, 20, 3, 0.260, 58, 65, 228, 0.22, 0.12),
        _make_hitter("Nick Castellanos", "PHI", "OF", 540, 18, 3, 0.258, 68, 72, 230, 0.10, 0.08),
        _make_hitter("Eugenio Suarez", "ARI", "3B", 480, 20, 3, 0.240, 62, 72, 232, 0.12, 0.12),
        _make_hitter("Tyler Stephenson", "CIN", "C,1B", 420, 14, 2, 0.265, 50, 58, 234, 0.15, 0.22),
        _make_hitter("Maikel Garcia", "KC", "SS,3B", 530, 8, 22, 0.258, 68, 48, 236, 0.15, 0.08),
        _make_hitter("Jonah Heim", "TEX", "C", 460, 15, 2, 0.248, 55, 62, 238, 0.12, 0.10),
        _make_hitter("Jose Siri", "TB", "OF", 420, 18, 15, 0.235, 58, 55, 240, 0.20, 0.12),
        _make_hitter("Nick Gonzales", "PIT", "2B", 450, 14, 10, 0.262, 58, 52, 242, 0.28, 0.12),
        _make_hitter("MJ Melendez", "KC", "C,OF", 420, 15, 5, 0.235, 52, 55, 244, 0.22, 0.12),
        _make_hitter("Logan O'Hoppe", "LAA", "C", 450, 18, 2, 0.250, 55, 62, 246, 0.20, 0.12),
        _make_hitter("Nico Hoerner", "CHC", "2B,SS", 520, 8, 18, 0.268, 68, 48, 248, 0.10, 0.12),
        _make_hitter("Mark Vientos", "NYM", "3B,1B", 480, 22, 2, 0.252, 62, 72, 250, 0.25, 0.10),
        _make_hitter("Lourdes Gurriel Jr.", "ARI", "OF", 480, 18, 8, 0.265, 62, 68, 252, 0.12, 0.10),
        _make_hitter("Byron Buxton", "MIN", "OF", 380, 22, 12, 0.248, 58, 58, 254, 0.22, 0.42),
        _make_hitter("Daulton Varsho", "TOR", "C,OF", 520, 18, 12, 0.242, 72, 65, 256, 0.15, 0.10),
        # ==================================================================
        # DEEP / BENCH TIER  (ADP 258+)
        # ==================================================================
        _make_hitter("Brendan Rodgers", "COL", "2B", 420, 14, 5, 0.260, 52, 55, 258, 0.18, 0.18),
        _make_hitter("Yoan Moncada", "CLE", "3B", 380, 12, 5, 0.258, 48, 48, 260, 0.22, 0.35),
        _make_hitter("Max Kepler", "MIN", "OF", 400, 15, 5, 0.248, 55, 55, 262, 0.12, 0.12),
        _make_hitter("Mitch Garver", "SEA", "C,DH", 380, 18, 2, 0.245, 48, 58, 264, 0.15, 0.25),
        _make_hitter("Trent Grisham", "NYY", "OF", 380, 12, 10, 0.235, 50, 42, 266, 0.18, 0.12),
        _make_hitter("Sean Murphy", "ATL", "C", 420, 18, 1, 0.245, 52, 62, 268, 0.12, 0.22),
        _make_hitter("Carlos Santana", "MIN", "1B,DH", 450, 18, 2, 0.235, 58, 65, 270, 0.10, 0.12),
        _make_hitter("Leody Taveras", "TEX", "OF", 420, 8, 18, 0.252, 55, 42, 272, 0.18, 0.10),
        _make_hitter("Drew Waters", "KC", "OF", 380, 10, 15, 0.248, 48, 40, 274, 0.32, 0.15),
        _make_hitter("Gabriel Moreno", "ARI", "C", 420, 10, 5, 0.272, 48, 52, 276, 0.18, 0.12),
        _make_hitter("Eddie Rosario", "WSH", "OF", 400, 15, 5, 0.252, 48, 55, 278, 0.15, 0.15),
        _make_hitter("Whit Merrifield", "ATL", "2B,OF", 400, 8, 12, 0.258, 52, 42, 280, 0.12, 0.15),
        _make_hitter("Taylor Ward", "LAA", "OF", 430, 15, 5, 0.255, 55, 55, 282, 0.15, 0.15),
        _make_hitter("Yainer Diaz", "HOU", "C,DH", 460, 15, 2, 0.262, 52, 62, 284, 0.15, 0.10),
        _make_hitter("Adolis Garcia", "TEX", "OF", 500, 22, 12, 0.242, 68, 72, 286, 0.15, 0.15),
        _make_hitter("Mitch Haniger", "SEA", "OF,DH", 420, 18, 3, 0.248, 55, 60, 288, 0.15, 0.22),
        _make_hitter("Tommy Pham", "STL", "OF,DH", 380, 12, 8, 0.252, 48, 48, 290, 0.15, 0.15),
        _make_hitter("Jonathan India", "CIN", "2B", 480, 12, 10, 0.252, 62, 52, 292, 0.15, 0.12),
        _make_hitter("Patrick Bailey", "SF", "C", 420, 12, 3, 0.255, 48, 52, 294, 0.18, 0.10),
        _make_hitter("Jorge Soler", "ATL", "OF,DH", 480, 22, 3, 0.240, 65, 72, 296, 0.15, 0.12),
        _make_hitter("Austin Wells", "NYY", "C,DH", 420, 15, 3, 0.255, 52, 55, 298, 0.22, 0.12),
        _make_hitter("Jace Jung", "DET", "2B,3B", 380, 14, 3, 0.252, 48, 52, 300, 0.35, 0.15),
        _make_hitter("Luis Arraez", "SD", "1B,2B", 520, 5, 2, 0.300, 62, 48, 302, 0.08, 0.12),
        _make_hitter("Jake McCarthy", "ARI", "OF", 420, 10, 18, 0.258, 55, 42, 304, 0.18, 0.12),
        _make_hitter("Enrique Hernandez", "LAD", "2B,SS,OF", 380, 10, 5, 0.245, 48, 42, 306, 0.12, 0.12),
        _make_hitter("Miguel Vargas", "LAD", "3B,1B", 380, 12, 5, 0.255, 45, 48, 308, 0.28, 0.12),
        _make_hitter("Harrison Bader", "NYM", "OF", 400, 12, 12, 0.248, 52, 45, 310, 0.15, 0.15),
        # ==================================================================
        # EXTRA DEPTH  (ADP 312+)
        # ==================================================================
        _make_hitter("Andrew McCutchen", "KC", "OF,DH", 420, 14, 5, 0.248, 52, 55, 312, 0.12, 0.18),
        _make_hitter("Ty France", "CIN", "1B,2B", 450, 12, 2, 0.262, 52, 58, 314, 0.12, 0.12),
        _make_hitter("Myles Straw", "CLE", "OF", 400, 3, 20, 0.255, 52, 32, 316, 0.15, 0.10),
        _make_hitter("Adam Frazier", "KC", "2B,OF", 420, 8, 8, 0.262, 52, 45, 318, 0.12, 0.10),
        _make_hitter("Nolan Jones", "COL", "OF,1B", 450, 20, 8, 0.245, 62, 65, 320, 0.25, 0.18),
        _make_hitter("Nate Lowe", "ARI", "1B", 450, 18, 2, 0.258, 55, 65, 322, 0.12, 0.10),
        _make_hitter("Luis Rengifo", "LAA", "2B,3B,SS", 480, 14, 10, 0.262, 60, 55, 324, 0.15, 0.12),
        _make_hitter("Wenceel Perez", "DET", "2B,OF", 450, 10, 15, 0.265, 58, 45, 326, 0.25, 0.10),
        _make_hitter("Isiah Kiner-Falefa", "PIT", "SS,3B", 450, 8, 12, 0.262, 55, 45, 328, 0.12, 0.08),
        _make_hitter("Rowdy Tellez", "PIT", "1B,DH", 420, 18, 1, 0.248, 48, 58, 330, 0.15, 0.15),
        _make_hitter("Josh Smith", "TEX", "3B,OF", 420, 10, 10, 0.252, 55, 45, 332, 0.22, 0.12),
        _make_hitter("Matt Chapman", "SF", "3B", 520, 22, 3, 0.242, 72, 78, 334, 0.10, 0.10),
        _make_hitter("Jose Caballero", "TB", "SS,2B", 400, 8, 22, 0.242, 55, 35, 336, 0.25, 0.12),
        _make_hitter("Shea Langeliers", "OAK", "C", 430, 18, 2, 0.232, 48, 58, 338, 0.18, 0.12),
        _make_hitter("Dylan Crews", "WSH", "OF", 400, 12, 10, 0.258, 52, 48, 340, 0.38, 0.12),
        _make_hitter("Robert Hassell III", "WSH", "OF", 380, 10, 8, 0.260, 48, 42, 342, 0.40, 0.15),
        _make_hitter("Coby Mayo", "BAL", "3B,1B", 400, 18, 3, 0.248, 52, 58, 344, 0.38, 0.15),
        _make_hitter("Junior Caminero", "TB", "3B", 380, 15, 5, 0.262, 48, 52, 346, 0.35, 0.15),
        _make_hitter("Pete Crow-Armstrong", "CHC", "OF", 420, 10, 20, 0.248, 55, 42, 348, 0.30, 0.10),
        _make_hitter("Travis Swaggerty", "PIT", "OF", 380, 8, 12, 0.252, 45, 38, 350, 0.30, 0.15),
        _make_hitter("Sal Frelick", "MIL", "OF", 460, 8, 15, 0.272, 62, 48, 352, 0.20, 0.08),
        _make_hitter("TJ Friedl", "CIN", "OF", 420, 12, 18, 0.260, 58, 45, 354, 0.18, 0.18),
        _make_hitter("Triston Casas", "BOS", "1B", 480, 22, 2, 0.250, 65, 72, 356, 0.20, 0.22),
        _make_hitter("Anthony Rizzo", "NYY", "1B", 430, 18, 2, 0.248, 52, 62, 358, 0.12, 0.22),
        _make_hitter("Jeff McNeil", "NYM", "2B,OF", 430, 8, 3, 0.272, 48, 48, 360, 0.12, 0.12),
        _make_hitter("Michael Busch", "CHC", "1B,2B", 450, 20, 5, 0.245, 58, 65, 362, 0.25, 0.12),
        _make_hitter("Garrett Mitchell", "MIL", "OF", 380, 12, 18, 0.248, 50, 42, 364, 0.30, 0.30),
        _make_hitter("Jung Hoo Lee", "SF", "OF", 480, 10, 8, 0.278, 62, 52, 366, 0.18, 0.18),
        _make_hitter("Andy Pages", "LAD", "OF", 400, 18, 5, 0.242, 52, 55, 368, 0.30, 0.12),
        _make_hitter("Emmanuel Rivera", "MIA", "3B,1B", 380, 14, 2, 0.252, 42, 50, 370, 0.18, 0.12),
        _make_hitter("Bobby Dalbec", "WSH", "1B", 380, 18, 2, 0.228, 45, 52, 372, 0.22, 0.18),
        _make_hitter("Lenyn Sosa", "CHW", "2B,SS", 400, 12, 5, 0.255, 45, 48, 374, 0.28, 0.12),
        _make_hitter("Heston Kjerstad", "BAL", "OF,DH", 380, 15, 3, 0.258, 45, 50, 376, 0.35, 0.18),
        _make_hitter("Jasson Dominguez", "NYY", "OF", 360, 14, 12, 0.252, 48, 42, 378, 0.40, 0.25),
        _make_hitter("Tyler Soderstrom", "OAK", "C,1B", 420, 18, 2, 0.242, 48, 55, 380, 0.30, 0.12),
        _make_hitter("Joey Ortiz", "MIL", "3B,SS", 450, 12, 5, 0.258, 52, 52, 382, 0.22, 0.10),
        _make_hitter("Masyn Winn", "STL", "SS", 480, 12, 15, 0.255, 58, 48, 384, 0.28, 0.10),
        _make_hitter("Brooks Lee", "MIN", "SS,3B", 400, 12, 5, 0.262, 48, 50, 386, 0.35, 0.12),
        _make_hitter("Davis Schneider", "TOR", "2B,OF", 380, 15, 3, 0.242, 48, 48, 388, 0.30, 0.15),
        _make_hitter("Nolan Schanuel", "LAA", "1B", 450, 10, 3, 0.265, 52, 48, 390, 0.22, 0.10),
        _make_hitter("Yonny Hernandez", "ARI", "2B,SS", 360, 2, 18, 0.258, 42, 25, 392, 0.22, 0.10),
        _make_hitter("Ryan Noda", "OAK", "1B,OF", 400, 18, 2, 0.232, 48, 55, 394, 0.18, 0.12),
    ]


# ---------------------------------------------------------------------------
# Pitcher projections (~80 players)
# ---------------------------------------------------------------------------


def get_pitcher_projections() -> list[dict]:
    """Return ~80 pitcher projections for 2026 drafts."""
    return [
        # ==================================================================
        # ELITE SP  (ADP 15-60)
        # ==================================================================
        _make_pitcher("Tarik Skubal", "DET", "SP", 195, 16, 0, 230, 2.70, 1.02, 15, 0.10, 0.12, gs=32),
        _make_pitcher("Paul Skenes", "PIT", "SP", 190, 15, 0, 225, 2.80, 1.05, 18, 0.28, 0.15, gs=31),
        _make_pitcher("Corbin Burnes", "ARI", "SP", 200, 15, 0, 210, 2.90, 1.08, 21, 0.08, 0.10, gs=33),
        _make_pitcher("Zack Wheeler", "PHI", "SP", 195, 14, 0, 215, 2.85, 1.05, 23, 0.08, 0.12, gs=32),
        _make_pitcher("Gerrit Cole", "NYY", "SP", 185, 14, 0, 220, 2.95, 1.08, 25, 0.10, 0.15, gs=31),
        _make_pitcher("Chris Sale", "ATL", "SP", 180, 15, 0, 210, 2.80, 1.02, 27, 0.10, 0.22, gs=30),
        _make_pitcher("Logan Webb", "SF", "SP", 200, 14, 0, 185, 3.05, 1.12, 29, 0.08, 0.08, gs=33),
        _make_pitcher("Cole Ragans", "KC", "SP", 185, 13, 0, 210, 3.10, 1.10, 31, 0.18, 0.12, gs=31),
        _make_pitcher("Dylan Cease", "SD", "SP", 185, 13, 0, 205, 3.25, 1.15, 33, 0.12, 0.10, gs=31),
        _make_pitcher("Framber Valdez", "HOU", "SP", 195, 14, 0, 185, 3.15, 1.12, 35, 0.10, 0.10, gs=32),
        _make_pitcher("Tyler Glasnow", "LAD", "SP", 170, 12, 0, 210, 3.00, 1.05, 37, 0.18, 0.40, gs=28),
        _make_pitcher("Max Fried", "NYY", "SP", 185, 14, 0, 180, 3.10, 1.10, 39, 0.10, 0.12, gs=31),
        _make_pitcher("Aaron Nola", "PHI", "SP", 195, 13, 0, 200, 3.30, 1.12, 41, 0.08, 0.08, gs=32),
        _make_pitcher("Yoshinobu Yamamoto", "LAD", "SP", 170, 12, 0, 185, 3.05, 1.08, 43, 0.22, 0.18, gs=28),
        _make_pitcher("Spencer Strider", "ATL", "SP", 140, 10, 0, 180, 2.90, 1.00, 45, 0.35, 0.50, gs=24),
        _make_pitcher("Garrett Crochet", "BOS", "SP", 180, 12, 0, 220, 3.20, 1.12, 47, 0.25, 0.18, gs=30),
        _make_pitcher("Hunter Greene", "CIN", "SP", 180, 12, 0, 210, 3.30, 1.12, 49, 0.20, 0.15, gs=30),
        _make_pitcher("Tanner Bibee", "CLE", "SP", 185, 13, 0, 195, 3.25, 1.10, 51, 0.18, 0.10, gs=31),
        _make_pitcher("Seth Lugo", "KC", "SP", 185, 14, 0, 175, 3.30, 1.15, 53, 0.12, 0.10, gs=31),
        _make_pitcher("Logan Gilbert", "SEA", "SP", 190, 13, 0, 190, 3.35, 1.12, 55, 0.12, 0.10, gs=32),
        # ==================================================================
        # MID SP  (ADP 57-119)
        # ==================================================================
        _make_pitcher("Luis Castillo", "SEA", "SP", 185, 12, 0, 185, 3.50, 1.18, 57, 0.10, 0.10, gs=31),
        _make_pitcher("Sonny Gray", "STL", "SP", 180, 12, 0, 180, 3.40, 1.15, 59, 0.12, 0.15, gs=30),
        _make_pitcher("Pablo Lopez", "MIN", "SP", 180, 12, 0, 180, 3.60, 1.15, 61, 0.12, 0.12, gs=30),
        _make_pitcher("Zac Gallen", "ARI", "SP", 175, 12, 0, 180, 3.45, 1.15, 63, 0.12, 0.18, gs=29),
        _make_pitcher("Joe Ryan", "MIN", "SP", 170, 11, 0, 175, 3.55, 1.12, 65, 0.15, 0.12, gs=28),
        _make_pitcher("Mitch Keller", "PIT", "SP", 185, 12, 0, 175, 3.60, 1.18, 67, 0.12, 0.10, gs=31),
        _make_pitcher("Bailey Ober", "MIN", "SP", 175, 11, 0, 175, 3.50, 1.10, 69, 0.15, 0.10, gs=29),
        _make_pitcher("Cody Bradford", "TEX", "SP", 170, 11, 0, 155, 3.45, 1.12, 71, 0.22, 0.12, gs=28),
        _make_pitcher("Michael King", "SD", "SP", 175, 11, 0, 180, 3.55, 1.18, 73, 0.18, 0.15, gs=29),
        _make_pitcher("Clarke Schmidt", "NYY", "SP", 165, 10, 0, 160, 3.60, 1.18, 75, 0.18, 0.18, gs=28),
        _make_pitcher("Jesus Luzardo", "MIA", "SP", 155, 10, 0, 175, 3.50, 1.15, 77, 0.18, 0.28, gs=26),
        _make_pitcher("Bryce Miller", "SEA", "SP", 175, 11, 0, 165, 3.65, 1.18, 79, 0.18, 0.12, gs=29),
        _make_pitcher("Ranger Suarez", "PHI", "SP", 175, 12, 0, 160, 3.40, 1.18, 81, 0.12, 0.12, gs=29),
        _make_pitcher("Nick Lodolo", "CIN", "SP", 160, 10, 0, 170, 3.70, 1.20, 83, 0.22, 0.22, gs=27),
        _make_pitcher("Ronel Blanco", "HOU", "SP", 170, 11, 0, 165, 3.55, 1.15, 85, 0.22, 0.12, gs=28),
        _make_pitcher("MacKenzie Gore", "WSH", "SP", 165, 10, 0, 175, 3.80, 1.22, 87, 0.22, 0.18, gs=28),
        _make_pitcher("Jared Jones", "PIT", "SP", 155, 10, 0, 170, 3.65, 1.18, 89, 0.30, 0.18, gs=26),
        _make_pitcher("Max Meyer", "MIA", "SP", 145, 9, 0, 155, 3.70, 1.18, 91, 0.30, 0.25, gs=24),
        _make_pitcher("Tyler Anderson", "LAA", "SP", 170, 10, 0, 145, 3.90, 1.25, 93, 0.10, 0.10, gs=28),
        _make_pitcher("Jordan Montgomery", "ARI", "SP", 170, 10, 0, 150, 3.85, 1.22, 95, 0.12, 0.12, gs=28),
        _make_pitcher("Reid Detmers", "LAA", "SP", 155, 9, 0, 160, 3.95, 1.25, 97, 0.22, 0.15, gs=26),
        _make_pitcher("George Kirby", "SEA", "SP", 180, 12, 0, 170, 3.40, 1.12, 99, 0.12, 0.10, gs=30),
        _make_pitcher("Spencer Schwellenbach", "ATL", "SP", 165, 10, 0, 165, 3.55, 1.15, 101, 0.28, 0.12, gs=28),
        _make_pitcher("Gavin Williams", "CLE", "SP", 150, 9, 0, 160, 3.75, 1.20, 103, 0.30, 0.22, gs=25),
        _make_pitcher("Grayson Rodriguez", "BAL", "SP", 170, 11, 0, 175, 3.65, 1.18, 105, 0.22, 0.15, gs=28),
        _make_pitcher("Brady Singer", "KC", "SP", 170, 10, 0, 155, 3.80, 1.22, 107, 0.15, 0.12, gs=28),
        _make_pitcher("Shane Baz", "TB", "SP", 130, 8, 0, 140, 3.60, 1.15, 109, 0.35, 0.35, gs=22),
        _make_pitcher("Justin Steele", "CHC", "SP", 170, 11, 0, 165, 3.55, 1.18, 111, 0.15, 0.18, gs=28),
        _make_pitcher("Jack Flaherty", "DET", "SP", 170, 10, 0, 175, 3.80, 1.20, 113, 0.18, 0.18, gs=28),
        _make_pitcher("Tobias Myers", "MIL", "SP", 155, 9, 0, 140, 3.90, 1.22, 119, 0.25, 0.12, gs=26),
        # ==================================================================
        # ELITE RP / CLOSERS  (ADP 117-149)
        # ==================================================================
        _make_pitcher("Emmanuel Clase", "CLE", "RP", 72, 4, 42, 80, 2.10, 0.92, 117, 0.08, 0.08, gs=0),
        _make_pitcher("Josh Hader", "HOU", "RP", 65, 3, 38, 85, 2.40, 0.95, 121, 0.10, 0.10, gs=0),
        _make_pitcher("Ryan Helsley", "STL", "RP", 68, 4, 40, 90, 2.20, 0.90, 125, 0.12, 0.12, gs=0),
        _make_pitcher("Devin Williams", "NYY", "RP", 62, 3, 35, 85, 2.30, 0.95, 129, 0.15, 0.22, gs=0),
        _make_pitcher("Edwin Diaz", "NYM", "RP", 60, 3, 32, 80, 2.80, 1.05, 133, 0.18, 0.28, gs=0),
        _make_pitcher("Raisel Iglesias", "ATL", "RP", 65, 3, 35, 72, 2.60, 1.00, 137, 0.10, 0.10, gs=0),
        _make_pitcher("Mason Miller", "OAK", "RP", 62, 3, 32, 88, 2.50, 0.98, 141, 0.25, 0.18, gs=0),
        _make_pitcher("Robert Suarez", "SD", "RP", 65, 3, 35, 72, 2.70, 1.02, 145, 0.12, 0.12, gs=0),
        _make_pitcher("Andres Munoz", "SEA", "RP", 62, 3, 30, 82, 2.55, 1.00, 149, 0.15, 0.15, gs=0),
        # ==================================================================
        # MID RP / CLOSERS  (ADP 155-200)
        # ==================================================================
        _make_pitcher("Kenley Jansen", "BOS", "RP", 60, 3, 28, 65, 3.20, 1.15, 155, 0.12, 0.18, gs=0),
        _make_pitcher("Tanner Scott", "SD", "RP", 62, 3, 30, 75, 3.00, 1.12, 159, 0.15, 0.12, gs=0),
        _make_pitcher("Carlos Estevez", "PHI", "RP", 60, 3, 28, 65, 3.10, 1.12, 163, 0.15, 0.12, gs=0),
        _make_pitcher("Jeff Hoffman", "PHI", "RP", 65, 4, 25, 78, 2.80, 1.08, 165, 0.15, 0.12, gs=0),
        _make_pitcher("Pete Fairbanks", "TB", "RP", 55, 3, 25, 65, 3.15, 1.10, 169, 0.18, 0.25, gs=0),
        _make_pitcher("Clay Holmes", "NYM", "RP", 62, 3, 28, 60, 3.30, 1.20, 171, 0.15, 0.15, gs=0),
        _make_pitcher("Jhoan Duran", "MIN", "RP", 65, 4, 25, 78, 3.00, 1.08, 175, 0.18, 0.15, gs=0),
        _make_pitcher("Daniel Hudson", "LAD", "RP", 55, 3, 22, 55, 3.40, 1.18, 177, 0.12, 0.22, gs=0),
        _make_pitcher("Alexis Diaz", "CIN", "RP", 60, 3, 28, 72, 3.20, 1.12, 181, 0.15, 0.12, gs=0),
        _make_pitcher("Paul Sewald", "ARI", "RP", 58, 3, 22, 62, 3.40, 1.15, 183, 0.12, 0.15, gs=0),
        _make_pitcher("Evan Phillips", "LAD", "RP", 58, 3, 22, 62, 3.30, 1.12, 187, 0.15, 0.15, gs=0),
        # ==================================================================
        # LATE RP  (ADP 189-275)
        # ==================================================================
        _make_pitcher("Jordan Romano", "TOR", "RP", 55, 2, 22, 60, 3.50, 1.18, 189, 0.18, 0.30, gs=0),
        _make_pitcher("Craig Kimbrel", "BAL", "RP", 55, 3, 20, 65, 3.60, 1.20, 193, 0.12, 0.22, gs=0),
        _make_pitcher("David Bednar", "PIT", "RP", 58, 3, 25, 68, 3.30, 1.15, 195, 0.15, 0.18, gs=0),
        _make_pitcher("Camilo Doval", "SF", "RP", 60, 3, 25, 70, 3.40, 1.18, 200, 0.18, 0.15, gs=0),
        _make_pitcher("Yennier Cano", "BAL", "RP", 60, 4, 12, 55, 3.20, 1.12, 210, 0.18, 0.12, gs=0),
        _make_pitcher("Hunter Harvey", "KC", "RP", 55, 3, 18, 60, 3.50, 1.18, 220, 0.18, 0.22, gs=0),
        _make_pitcher("Aroldis Chapman", "PIT", "RP", 55, 3, 18, 70, 3.60, 1.25, 225, 0.15, 0.22, gs=0),
        _make_pitcher("AJ Minter", "ATL", "RP", 55, 3, 15, 58, 3.55, 1.20, 235, 0.15, 0.18, gs=0),
        _make_pitcher("Joe Jimenez", "CLE", "RP", 55, 3, 15, 55, 3.70, 1.22, 243, 0.15, 0.15, gs=0),
        _make_pitcher("Bryan Abreu", "HOU", "RP", 58, 3, 15, 72, 3.40, 1.15, 255, 0.18, 0.15, gs=0),
        _make_pitcher("Kirby Yates", "TEX", "RP", 55, 3, 22, 58, 3.40, 1.15, 275, 0.15, 0.18, gs=0),
    ]
