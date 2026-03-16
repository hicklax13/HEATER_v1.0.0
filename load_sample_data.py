"""Generate sample projection data for testing the draft tool.

Run this to populate the database with realistic sample data.
For the real draft, replace this with actual FanGraphs CSV imports.
"""

import logging

import numpy as np

from src.database import get_connection, init_db

logger = logging.getLogger(__name__)


def generate_sample_data():
    # Seed for reproducibility — inside function for idempotent results
    rng = np.random.default_rng(42)

    """Generate ~300 hitters and ~200 pitchers with realistic projections."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM projections")
    cursor.execute("DELETE FROM adp")
    cursor.execute("DELETE FROM players")
    # Clear Plan 3 and in-season tables if they exist
    for tbl in ("injury_history", "transactions", "league_rosters", "league_standings"):
        try:
            cursor.execute(f"DELETE FROM {tbl}")
        except Exception:
            pass
    conn.commit()

    # ── Hitters ──────────────────────────────────────────────────────

    hitter_archetypes = [
        # (name_prefix, positions, pa_range, hr_range, sb_range, avg_range, tier)
        # Elite
        ("Aaron Judge", "OF", (600, 650), (45, 55), (3, 8), (0.285, 0.310), 1),
        ("Shohei Ohtani", "Util", (580, 620), (40, 50), (15, 25), (0.280, 0.300), 1),
        ("Mookie Betts", "OF,SS", (580, 620), (25, 35), (12, 20), (0.280, 0.300), 1),
        ("Freddie Freeman", "1B", (600, 650), (22, 30), (5, 10), (0.290, 0.315), 1),
        ("Trea Turner", "SS", (560, 610), (20, 28), (25, 35), (0.275, 0.295), 1),
        ("Bobby Witt Jr.", "SS", (620, 660), (28, 35), (25, 40), (0.280, 0.300), 1),
        ("Elly De La Cruz", "SS", (560, 610), (22, 30), (55, 75), (0.255, 0.275), 1),
        ("Ronald Acuna Jr.", "OF", (540, 600), (20, 28), (40, 60), (0.270, 0.295), 2),
        ("Corey Seager", "SS", (560, 610), (28, 38), (2, 6), (0.275, 0.295), 2),
        ("Juan Soto", "OF", (600, 650), (30, 40), (3, 8), (0.275, 0.300), 1),
        ("Yordan Alvarez", "Util,OF", (560, 600), (30, 40), (1, 5), (0.280, 0.305), 2),
        ("Marcus Semien", "2B", (620, 660), (22, 30), (15, 25), (0.265, 0.285), 2),
        ("Jose Ramirez", "3B", (580, 620), (25, 33), (18, 28), (0.270, 0.290), 2),
        ("Rafael Devers", "3B", (580, 620), (28, 36), (1, 5), (0.275, 0.295), 2),
        ("Kyle Tucker", "OF", (560, 600), (25, 33), (18, 28), (0.275, 0.295), 2),
        ("Julio Rodriguez", "OF", (580, 620), (22, 30), (22, 35), (0.270, 0.290), 2),
        ("Gunnar Henderson", "SS,3B", (600, 640), (28, 36), (15, 25), (0.265, 0.285), 2),
        ("Matt Olson", "1B", (580, 620), (30, 42), (1, 4), (0.248, 0.268), 3),
        ("Pete Alonso", "1B", (560, 600), (35, 45), (1, 4), (0.245, 0.265), 3),
        ("Adolis Garcia", "OF", (560, 600), (28, 36), (18, 28), (0.248, 0.268), 3),
        ("William Contreras", "C", (520, 570), (18, 26), (1, 5), (0.275, 0.295), 3),
        ("Adley Rutschman", "C", (540, 580), (18, 25), (2, 6), (0.265, 0.285), 3),
        ("Salvador Perez", "C", (520, 560), (22, 32), (0, 3), (0.255, 0.275), 4),
        ("J.T. Realmuto", "C", (480, 530), (15, 22), (10, 18), (0.260, 0.280), 4),
        ("Cal Raleigh", "C", (480, 530), (25, 33), (1, 4), (0.230, 0.250), 4),
        ("Willy Adames", "SS", (560, 600), (25, 33), (15, 22), (0.248, 0.268), 3),
        ("Bo Bichette", "SS", (550, 600), (18, 25), (15, 25), (0.270, 0.290), 3),
        ("Ozzie Albies", "2B", (560, 600), (22, 30), (10, 18), (0.265, 0.285), 3),
        ("Jazz Chisholm Jr.", "3B,2B", (520, 570), (20, 28), (18, 28), (0.250, 0.270), 3),
        ("Christian Walker", "1B", (560, 600), (28, 36), (3, 8), (0.255, 0.275), 3),
        ("Anthony Santander", "OF", (560, 600), (30, 40), (2, 6), (0.255, 0.275), 3),
        ("Anthony Rizzo", "1B", (480, 530), (18, 26), (1, 4), (0.255, 0.275), 5),
        ("Tommy Edman", "2B,SS,OF", (520, 570), (10, 18), (25, 40), (0.260, 0.280), 4),
        ("Thairo Estrada", "2B,SS", (500, 550), (12, 18), (12, 20), (0.265, 0.285), 5),
        ("Byron Buxton", "OF", (400, 480), (25, 35), (10, 18), (0.248, 0.268), 4),
        ("Mike Trout", "OF", (380, 460), (25, 35), (3, 8), (0.260, 0.285), 4),
        ("Lars Nootbaar", "OF", (480, 540), (15, 22), (8, 15), (0.255, 0.275), 5),
        ("Jarren Duran", "OF", (580, 620), (15, 22), (30, 45), (0.275, 0.295), 3),
        ("Jackson Chourio", "OF", (560, 600), (18, 26), (15, 25), (0.270, 0.290), 3),
        ("Jackson Merrill", "OF", (560, 600), (20, 28), (10, 18), (0.270, 0.290), 3),
    ]

    # Generate additional filler hitters with real positions
    filler_hitters = {
        "Alex Bregman": "3B",
        "Nolan Arenado": "3B",
        "Manny Machado": "3B",
        "Xander Bogaerts": "SS",
        "Carlos Correa": "SS",
        "Tim Anderson": "SS",
        "Dansby Swanson": "SS",
        "Ketel Marte": "2B,OF",
        "Gleyber Torres": "2B",
        "Jonathan India": "2B",
        "Andres Gimenez": "2B",
        "Ha-Seong Kim": "SS,2B,3B",
        "Yandy Diaz": "1B,3B",
        "Vladimir Guerrero Jr.": "1B",
        "Spencer Torkelson": "1B",
        "Vinnie Pasquantino": "1B",
        "Rhys Hoskins": "1B",
        "Cody Bellinger": "OF,1B",
        "Bryce Harper": "1B",
        "Luis Robert Jr.": "OF",
        "Mike Yastrzemski": "OF",
        "Lane Thomas": "OF",
        "Bryan Reynolds": "OF",
        "Starling Marte": "OF",
        "Randy Arozarena": "OF",
        "George Springer": "OF",
        "Teoscar Hernandez": "OF",
        "Giancarlo Stanton": "OF",
        "Marcell Ozuna": "Util",
        "Austin Riley": "3B",
        "Max Muncy": "1B,3B",
        "Willson Contreras": "C",
        "Gabriel Moreno": "C",
        "MJ Melendez": "C",
        "Logan O'Hoppe": "C",
        "Patrick Bailey": "C",
        "Jonah Heim": "C",
        "Alejandro Kirk": "C",
        "Ryan McMahon": "2B,3B",
        "Brandon Lowe": "2B",
        "Jeff McNeil": "2B",
        "Nico Hoerner": "2B,SS",
        "CJ Abrams": "SS",
        "Masataka Yoshida": "OF",
        "Seiya Suzuki": "OF",
        "Fernando Tatis Jr.": "OF,SS",
        "Cedric Mullins": "OF",
        "Corbin Carroll": "OF",
        "Colton Cowser": "OF",
        "James Outman": "OF",
        "Michael Harris II": "OF",
        "Riley Greene": "OF",
        "Evan Carter": "OF",
        "Wyatt Langford": "OF",
        "Jordan Walker": "OF",
        "Spencer Steer": "1B,2B,3B",
        "Josh Lowe": "OF",
        "Willy Adames": "SS",
        "Tyler O'Neill": "OF",
        "Brendan Donovan": "2B,OF",
        "Isaac Paredes": "3B",
        "Christopher Morel": "OF,3B",
        "Maikel Garcia": "3B,SS",
        "Ezequiel Tovar": "SS",
        "Noelvi Marte": "3B",
        "Royce Lewis": "SS,3B",
        "Mookie Betts": "OF,SS",
        "Jake Cronenworth": "1B,2B",
    }
    filler_names = list(filler_hitters.keys())

    player_id = 1
    for name, pos, pa_r, hr_r, sb_r, avg_r, tier in hitter_archetypes:
        pa = rng.integers(*pa_r)
        avg = round(rng.uniform(*avg_r), 3)
        ab = int(pa * 0.87)
        h = int(ab * avg)
        hr = rng.integers(*hr_r)
        sb = rng.integers(*sb_r)
        r = int(hr * 1.8 + sb * 0.3 + rng.integers(30, 60))
        rbi = int(hr * 2.5 + rng.integers(15, 45))

        cursor.execute("INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, 1)", (name, "MLB", pos))
        pid = cursor.lastrowid
        cursor.execute(
            """
            INSERT INTO projections (player_id, system, pa, ab, h, r, hr, rbi, sb, avg)
            VALUES (?, 'blended', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (pid, pa, ab, h, r, hr, rbi, sb, avg),
        )

        adp = tier * 25 + rng.integers(-10, 10) + (player_id - 1) * 1.5
        cursor.execute("INSERT INTO adp (player_id, adp) VALUES (?, ?)", (pid, max(1, adp)))
        player_id += 1

    # Filler hitters
    seen_names = {name for name, *_ in hitter_archetypes}
    for name in filler_names:
        if name in seen_names:
            continue
        seen_names.add(name)
        pos = filler_hitters[name]
        pa = rng.integers(400, 600)
        avg = round(rng.uniform(0.235, 0.280), 3)
        ab = int(pa * 0.87)
        h = int(ab * avg)
        hr = rng.integers(8, 25)
        sb = rng.integers(2, 18)
        r = int(hr * 1.5 + sb * 0.3 + rng.integers(25, 50))
        rbi = int(hr * 2.2 + rng.integers(10, 35))

        cursor.execute("INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, 1)", (name, "MLB", pos))
        pid = cursor.lastrowid
        cursor.execute(
            """
            INSERT INTO projections (player_id, system, pa, ab, h, r, hr, rbi, sb, avg)
            VALUES (?, 'blended', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (pid, pa, ab, h, r, hr, rbi, sb, avg),
        )

        adp = 80 + player_id * 1.8 + rng.integers(-15, 15)
        cursor.execute("INSERT INTO adp (player_id, adp) VALUES (?, ?)", (pid, max(1, adp)))
        player_id += 1

    # ── Pitchers ─────────────────────────────────────────────────────

    pitcher_data = [
        # (name, pos, ip_range, w_range, k_range, era_range, whip_range, sv_range, tier)
        # Elite SP
        ("Zack Wheeler", "SP", (180, 210), (14, 18), (200, 240), (2.60, 3.10), (0.98, 1.10), (0, 0), 1),
        ("Spencer Strider", "SP", (160, 200), (12, 17), (220, 260), (2.50, 3.00), (0.95, 1.08), (0, 0), 1),
        ("Gerrit Cole", "SP", (175, 205), (13, 17), (210, 250), (2.70, 3.20), (0.98, 1.12), (0, 0), 1),
        ("Corbin Burnes", "SP", (180, 210), (12, 16), (190, 230), (2.80, 3.30), (1.00, 1.15), (0, 0), 1),
        ("Dylan Cease", "SP", (170, 200), (11, 15), (200, 240), (3.10, 3.60), (1.10, 1.25), (0, 0), 2),
        ("Logan Webb", "SP", (180, 210), (12, 16), (160, 200), (2.90, 3.40), (1.05, 1.18), (0, 0), 2),
        ("Tarik Skubal", "SP", (170, 200), (14, 18), (200, 240), (2.60, 3.10), (0.95, 1.10), (0, 0), 1),
        ("Zach Eflin", "SP", (170, 195), (12, 16), (160, 190), (3.10, 3.60), (1.05, 1.18), (0, 0), 3),
        ("Framber Valdez", "SP", (170, 200), (11, 15), (160, 200), (3.00, 3.50), (1.10, 1.25), (0, 0), 2),
        ("Max Fried", "SP", (170, 200), (12, 16), (160, 195), (2.80, 3.30), (1.05, 1.18), (0, 0), 2),
        ("Tyler Glasnow", "SP", (150, 185), (10, 14), (190, 230), (2.80, 3.40), (0.98, 1.12), (0, 0), 2),
        ("Chris Sale", "SP", (155, 185), (12, 16), (180, 220), (2.80, 3.30), (1.00, 1.15), (0, 0), 2),
        ("Pablo Lopez", "SP", (170, 195), (11, 15), (170, 210), (3.20, 3.70), (1.08, 1.22), (0, 0), 3),
        ("Sonny Gray", "SP", (160, 190), (11, 15), (170, 210), (3.10, 3.60), (1.08, 1.22), (0, 0), 3),
        ("Yoshinobu Yamamoto", "SP", (150, 180), (10, 14), (165, 200), (3.00, 3.50), (1.00, 1.15), (0, 0), 2),
        ("Kevin Gausman", "SP", (170, 195), (11, 15), (185, 220), (3.20, 3.70), (1.08, 1.22), (0, 0), 3),
        ("Luis Castillo", "SP", (170, 195), (10, 14), (180, 220), (3.30, 3.80), (1.12, 1.25), (0, 0), 3),
        ("Shota Imanaga", "SP", (155, 180), (10, 14), (160, 195), (3.00, 3.50), (1.02, 1.15), (0, 0), 3),
        ("Joe Ryan", "SP", (165, 190), (11, 14), (170, 205), (3.20, 3.70), (1.05, 1.18), (0, 0), 3),
        ("Tanner Houck", "SP", (165, 190), (11, 14), (160, 195), (3.20, 3.70), (1.10, 1.22), (0, 0), 3),
        # Mid-tier SP
        ("MacKenzie Gore", "SP", (155, 180), (9, 13), (160, 195), (3.40, 3.90), (1.15, 1.28), (0, 0), 4),
        ("Ranger Suarez", "SP", (160, 185), (10, 14), (145, 175), (3.20, 3.70), (1.10, 1.22), (0, 0), 3),
        ("Michael King", "SP", (155, 180), (9, 13), (165, 195), (3.30, 3.80), (1.12, 1.25), (0, 0), 4),
        ("Jared Jones", "SP", (140, 170), (8, 12), (155, 190), (3.40, 4.00), (1.15, 1.30), (0, 0), 4),
        ("Hunter Brown", "SP", (155, 180), (9, 13), (170, 205), (3.50, 4.10), (1.18, 1.32), (0, 0), 4),
        ("Garrett Crochet", "SP", (155, 185), (9, 13), (190, 230), (3.30, 3.80), (1.10, 1.22), (0, 0), 3),
        ("Paul Skenes", "SP", (150, 180), (10, 14), (190, 230), (2.80, 3.40), (0.98, 1.12), (0, 0), 2),
        # Elite RP/Closers
        ("Emmanuel Clase", "RP", (60, 72), (3, 6), (55, 70), (2.00, 2.80), (0.85, 1.00), (35, 45), 2),
        ("Josh Hader", "RP", (55, 68), (3, 6), (70, 90), (2.20, 3.00), (0.90, 1.05), (30, 42), 3),
        ("Devin Williams", "RP", (55, 68), (3, 6), (75, 95), (1.80, 2.60), (0.80, 0.98), (28, 40), 3),
        ("Ryan Helsley", "RP", (58, 70), (3, 6), (65, 85), (2.00, 2.80), (0.85, 1.00), (32, 42), 3),
        ("Kenley Jansen", "RP", (55, 68), (3, 5), (60, 78), (2.80, 3.60), (1.00, 1.15), (28, 38), 4),
        ("Andres Munoz", "RP", (58, 70), (3, 6), (72, 90), (2.20, 3.00), (0.88, 1.02), (30, 40), 3),
        ("Robert Suarez", "RP", (55, 68), (3, 5), (58, 72), (2.50, 3.30), (0.95, 1.10), (28, 38), 4),
        ("Pete Fairbanks", "RP", (50, 65), (3, 5), (55, 70), (2.60, 3.40), (0.98, 1.12), (25, 35), 4),
        ("Tanner Scott", "RP", (58, 70), (3, 6), (70, 88), (2.80, 3.60), (1.05, 1.18), (25, 35), 4),
        ("Carlos Estevez", "RP", (55, 68), (3, 5), (58, 72), (2.80, 3.60), (1.00, 1.15), (25, 35), 5),
        ("Mason Miller", "RP", (55, 68), (3, 5), (75, 95), (2.20, 3.00), (0.85, 1.00), (28, 38), 3),
        ("Clay Holmes", "RP", (55, 68), (3, 5), (55, 68), (3.00, 3.80), (1.10, 1.25), (22, 32), 5),
        ("Raisel Iglesias", "RP", (55, 68), (3, 5), (62, 78), (2.80, 3.60), (1.00, 1.15), (28, 38), 4),
    ]

    # Additional filler SP and RP
    filler_sp = [
        "Logan Gilbert",
        "Aaron Nola",
        "Freddy Peralta",
        "Shane Bieber",
        "Nick Lodolo",
        "Mitch Keller",
        "Bailey Ober",
        "Jake Irvin",
        "Cristian Javier",
        "Nathan Eovaldi",
        "Tyler Anderson",
        "Jordan Montgomery",
        "Nestor Cortes",
        "Reid Detmers",
        "Brayan Bello",
        "Gavin Stone",
        "Clarke Schmidt",
        "Spencer Schwellenbach",
        "Reese Olson",
        "Grayson Rodriguez",
        "Brady Singer",
        "Drew Thorpe",
        "Jack Flaherty",
        "Cole Ragans",
        "Seth Lugo",
        "George Kirby",
        "Tobias Myers",
        "JP Sears",
    ]
    filler_rp = [
        "Craig Kimbrel",
        "Jordan Romano",
        "David Bednar",
        "Felix Bautista",
        "Jhoan Duran",
        "Camilo Doval",
        "Paul Sewald",
        "Alexis Diaz",
        "Jason Adam",
        "Trevor Megill",
        "Yennier Cano",
        "Justin Martinez",
        "Ben Joyce",
        "Ryan Pressly",
        "Edwin Diaz",
        "A.J. Minter",
    ]

    pitcher_id_start = player_id
    for name, pos, ip_r, w_r, k_r, era_r, whip_r, sv_r, tier in pitcher_data:
        ip = round(rng.uniform(*ip_r), 1)
        w = rng.integers(*w_r)
        k = rng.integers(*k_r)
        era = round(rng.uniform(*era_r), 2)
        whip = round(rng.uniform(*whip_r), 2)
        sv = rng.integers(*sv_r) if sv_r[1] > 0 else 0
        er = int(round(era * ip / 9))
        total_baserunners = whip * ip
        bb = int(total_baserunners * 0.3)
        ha = int(total_baserunners * 0.7)

        cursor.execute("INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, 0)", (name, "MLB", pos))
        pid = cursor.lastrowid
        cursor.execute(
            """
            INSERT INTO projections (player_id, system, ip, w, sv, k, era, whip, er, bb_allowed, h_allowed)
            VALUES (?, 'blended', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (pid, ip, w, sv, k, era, whip, er, bb, ha),
        )

        adp = tier * 30 + rng.integers(-10, 10) + player_id * 0.8
        if pos == "RP":
            adp += 20
        cursor.execute("INSERT INTO adp (player_id, adp) VALUES (?, ?)", (pid, max(1, adp)))
        player_id += 1

    # Filler SP
    for name in filler_sp:
        ip = round(rng.uniform(140, 180), 1)
        w = rng.integers(8, 13)
        k = rng.integers(130, 185)
        era = round(rng.uniform(3.40, 4.20), 2)
        whip = round(rng.uniform(1.12, 1.30), 2)
        er = int(round(era * ip / 9))
        total_br = whip * ip
        bb = int(total_br * 0.3)
        ha = int(total_br * 0.7)

        cursor.execute("INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, 'SP', 0)", (name, "MLB"))
        pid = cursor.lastrowid
        cursor.execute(
            """
            INSERT INTO projections (player_id, system, ip, w, sv, k, era, whip, er, bb_allowed, h_allowed)
            VALUES (?, 'blended', ?, ?, 0, ?, ?, ?, ?, ?, ?)
        """,
            (pid, ip, w, k, era, whip, er, bb, ha),
        )

        adp = 100 + player_id * 1.5 + rng.integers(-10, 10)
        cursor.execute("INSERT INTO adp (player_id, adp) VALUES (?, ?)", (pid, max(1, adp)))
        player_id += 1

    # Filler RP
    for name in filler_rp:
        ip = round(rng.uniform(50, 70), 1)
        w = rng.integers(2, 5)
        k = rng.integers(50, 80)
        era = round(rng.uniform(2.80, 4.00), 2)
        whip = round(rng.uniform(0.95, 1.25), 2)
        sv = rng.integers(10, 30)
        er = int(round(era * ip / 9))
        total_br = whip * ip
        bb = int(total_br * 0.3)
        ha = int(total_br * 0.7)

        cursor.execute("INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, 'RP', 0)", (name, "MLB"))
        pid = cursor.lastrowid
        cursor.execute(
            """
            INSERT INTO projections (player_id, system, ip, w, sv, k, era, whip, er, bb_allowed, h_allowed)
            VALUES (?, 'blended', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (pid, ip, w, sv, k, era, whip, er, bb, ha),
        )

        adp = 140 + player_id * 1.2 + rng.integers(-10, 10)
        cursor.execute("INSERT INTO adp (player_id, adp) VALUES (?, ?)", (pid, max(1, adp)))
        player_id += 1

    conn.commit()

    # ── Birth dates & MLB IDs ──────────────────────────────────────
    # Get actual player IDs from the database (they may not start at 1)
    all_player_ids = [row[0] for row in cursor.execute("SELECT player_id FROM players").fetchall()]
    for pid in all_player_ids:
        birth_year = 2026 - rng.integers(22, 39)
        birth_month = rng.integers(1, 13)
        birth_day = rng.integers(1, 29)
        birth_date = f"{birth_year}-{birth_month:02d}-{birth_day:02d}"
        mlb_id = 600000 + pid * 7  # Fake but plausible MLB IDs
        try:
            cursor.execute(
                "UPDATE players SET birth_date = ?, mlb_id = ? WHERE player_id = ?",
                (birth_date, mlb_id, pid),
            )
        except Exception:
            pass  # Column may not exist if schema not updated

    # ── Injury history ─────────────────────────────────────────────
    # Generate 3 seasons of injury data for each player
    for pid in all_player_ids:
        for season in (2023, 2024, 2025):
            # Most players healthy (162 games), some injured
            if rng.random() < 0.25:  # 25% chance of significant injury season
                games_available = 162
                games_played = rng.integers(60, 140)
                il_stints = rng.integers(1, 4)
                il_days = rng.integers(15, 90)
            else:
                games_available = 162
                games_played = rng.integers(140, 163)
                il_stints = 0 if games_played > 155 else rng.integers(0, 2)
                il_days = 0 if il_stints == 0 else rng.integers(10, 30)
            try:
                cursor.execute(
                    """INSERT INTO injury_history
                       (player_id, season, games_played, games_available, il_stints, il_days)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (pid, season, int(games_played), games_available, int(il_stints), int(il_days)),
                )
            except Exception:
                pass  # Table may not exist

    # ── League rosters (12 teams, 23 players each) ────────────────
    # Distribute all players across 12 teams via snake draft order
    team_names = [
        "Team Hickey",
        "The Bombers",
        "Steal Squad",
        "Ace Hunters",
        "Diamond Dogs",
        "Bench Warmers",
        "Clutch Hitters",
        "Mound Masters",
        "Rally Caps",
        "Stat Padders",
        "Waiver Wire",
        "Pine Riders",
    ]
    roster_size = 23
    all_pids = [row[0] for row in cursor.execute("SELECT player_id FROM players ORDER BY player_id").fetchall()]
    # Assign players round-robin (snake style) to fill 23-man rosters
    team_rosters = {t: [] for t in team_names}
    idx = 0
    for pid in all_pids:
        team = team_names[idx % len(team_names)]
        if len(team_rosters[team]) < roster_size:
            team_rosters[team].append(pid)
        idx += 1

    for ti, team in enumerate(team_names):
        is_user = 1 if team == "Team Hickey" else 0
        team_idx = 0 if is_user else ti
        for pid in team_rosters[team]:
            cursor.execute(
                """INSERT INTO league_rosters (team_name, team_index, player_id, is_user_team)
                   VALUES (?, ?, ?, ?)""",
                (team, team_idx, pid, is_user),
            )

    # ── League standings (sample category totals) ────────────────
    cat_ranges = {
        "R": (280, 420),
        "HR": (80, 160),
        "RBI": (300, 450),
        "SB": (40, 120),
        "AVG": (0.240, 0.275),
        "W": (35, 65),
        "SV": (20, 55),
        "K": (550, 850),
        "ERA": (3.20, 4.60),
        "WHIP": (1.10, 1.35),
    }
    for team in team_names:
        for cat, (lo, hi) in cat_ranges.items():
            val = round(rng.uniform(lo, hi), 3)
            cursor.execute(
                """INSERT INTO league_standings (team_name, category, total)
                   VALUES (?, ?, ?)""",
                (team, cat, val),
            )

    conn.commit()
    conn.close()

    logger.info("Sample data loaded: %d total players (with injury history)", player_id - 1)
    logger.info("League rosters: %d teams x %d players", len(team_names), roster_size)
    return player_id - 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = generate_sample_data()
    logger.info("Done! %d players ready for drafting.", count)
