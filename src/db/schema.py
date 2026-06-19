"""Dialect-aware SQLAlchemy schema (B2.1) — the Postgres-side source of truth
the Alembic baseline builds. Mirrors init_db()'s SQLite schema exactly (enforced
by tests/test_db_alembic_baseline.py::test_schema_matches_init_db). init_db()
still owns the live SQLite path; this is additive.

NOTE: Real `alembic upgrade head` against a live Postgres is DEFERRED until a
PG instance exists (B2.2). This schema is verified via:
  1. SQLite upgrade via `alembic upgrade head` on a temp DB.
  2. PG-dialect DDL compilation tests (SERIAL / citext / partial WHERE).
  3. Schema-parity test against init_db()'s SQLite introspection.

FOREIGN KEYS are intentionally NOT declared here. The live SQLite app runs with
`PRAGMA foreign_keys = OFF` (get_connection never enables it), so init_db()'s ~14
`REFERENCES` edges are inert documentation today — enforcing them on Postgres would
add referential integrity the app has never had (new bootstrap-ordering + bulk-load
constraints, e.g. the user_id=0 default rows that have no users row). Enforcement is
a deliberate, separate hardening step — a later `op.create_foreign_key` migration
once the B2.5 data load is verified — not a silent baseline behavior change.
"""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import (
    PrimaryKeyConstraint as _PKC,
)
from sqlalchemy.dialects import postgresql

metadata = MetaData()

# ── 1. players ──────────────────────────────────────────────────────────────
# SERIAL on PG, INTEGER PK on SQLite (autoincrement handled by SA).
# All _safe_add_column additions folded in.
players = Table(
    "players",
    metadata,
    Column("player_id", Integer, primary_key=True),
    Column("name", Text, nullable=False),
    Column("team", Text),
    Column("positions", Text, nullable=False),
    Column("is_hitter", Integer, nullable=False, server_default=text("1")),
    Column("is_injured", Integer, nullable=False, server_default=text("0")),
    Column("injury_note", Text),
    # _safe_add_column additions
    Column("birth_date", Text),
    Column("mlb_id", Integer),
    Column("bats", Text),
    Column("throws", Text),
    Column("level", Text),  # NULL/"MLB"/"AAA"/"AA" — Wave 9 INFRA-F5
    Column("arbitration_eligible", Integer, server_default=text("0")),
    Column("fangraphs_id", Text),
    Column("yahoo_id", Text),
    Column("roster_type", Text, server_default=text("'active'")),
    Column("role_status", Text),
    Column("contract_details", Text),
    Column("spring_training_stats", Text),
    Column("depth_chart_role", Text),
    Column("contract_year", Integer, server_default=text("0")),
    Column("news_sentiment", Float),
    Column("lineup_slot", Integer),
    Column("spring_training_era", Float),
)

Index("idx_players_name", players.c.name)

# ── 2. projections ───────────────────────────────────────────────────────────
projections = Table(
    "projections",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("player_id", Integer, nullable=False),
    Column("system", Text, nullable=False),
    Column("forecast_season", Integer),
    # Hitting stats
    Column("pa", Integer, server_default=text("0")),
    Column("ab", Integer, server_default=text("0")),
    Column("h", Integer, server_default=text("0")),
    Column("r", Integer, server_default=text("0")),
    Column("hr", Integer, server_default=text("0")),
    Column("rbi", Integer, server_default=text("0")),
    Column("sb", Integer, server_default=text("0")),
    Column("avg", Float, server_default=text("0")),
    # Pitching stats
    Column("ip", Float, server_default=text("0")),
    Column("w", Integer, server_default=text("0")),
    Column("sv", Integer, server_default=text("0")),
    Column("k", Integer, server_default=text("0")),
    Column("era", Float, server_default=text("0")),
    Column("whip", Float, server_default=text("0")),
    Column("er", Integer, server_default=text("0")),
    Column("bb_allowed", Integer, server_default=text("0")),
    Column("h_allowed", Integer, server_default=text("0")),
    # _safe_add_column additions — birth_date/mlb_id
    Column("birth_date", Text),
    Column("mlb_id", Integer),
    # Phase 1 league format migration
    Column("obp", Float, server_default=text("0")),
    Column("l", Integer, server_default=text("0")),
    Column("bb", Integer, server_default=text("0")),
    Column("hbp", Integer, server_default=text("0")),
    Column("sf", Integer, server_default=text("0")),
    # Advanced pitcher metrics
    Column("fip", Float, server_default=text("0")),
    Column("xfip", Float, server_default=text("0")),
    Column("siera", Float, server_default=text("0")),
    # Statcast Stuff+/Location+/Pitching+
    Column("stuff_plus", Float),
    Column("location_plus", Float),
    Column("pitching_plus", Float),
)

Index("idx_projections_player", projections.c.player_id)
Index("idx_projections_system", projections.c.system)
Index("idx_projections_player_system", projections.c.player_id, projections.c.system)

# ── 3. adp ───────────────────────────────────────────────────────────────────
adp = Table(
    "adp",
    metadata,
    Column("player_id", Integer, primary_key=True),
    Column("yahoo_adp", Float),
    Column("fantasypros_adp", Float),
    Column("adp", Float, nullable=False),
    # _safe_add_column addition
    Column("nfbc_adp", Float),
)

# ── 4. season_stats ──────────────────────────────────────────────────────────
season_stats = Table(
    "season_stats",
    metadata,
    Column("player_id", Integer, nullable=False),
    Column("season", Integer, nullable=False, server_default=text("2026")),
    Column("pa", Integer, server_default=text("0")),
    Column("ab", Integer, server_default=text("0")),
    Column("h", Integer, server_default=text("0")),
    Column("r", Integer, server_default=text("0")),
    Column("hr", Integer, server_default=text("0")),
    Column("rbi", Integer, server_default=text("0")),
    Column("sb", Integer, server_default=text("0")),
    Column("avg", Float, server_default=text("0")),
    Column("ip", Float, server_default=text("0")),
    Column("w", Integer, server_default=text("0")),
    Column("sv", Integer, server_default=text("0")),
    Column("k", Integer, server_default=text("0")),
    Column("era", Float, server_default=text("0")),
    Column("whip", Float, server_default=text("0")),
    Column("er", Integer, server_default=text("0")),
    Column("bb_allowed", Integer, server_default=text("0")),
    Column("h_allowed", Integer, server_default=text("0")),
    Column("games_played", Integer, server_default=text("0")),
    Column("last_updated", Text),
    # Phase 1
    Column("obp", Float, server_default=text("0")),
    Column("l", Integer, server_default=text("0")),
    Column("bb", Integer, server_default=text("0")),
    Column("hbp", Integer, server_default=text("0")),
    Column("sf", Integer, server_default=text("0")),
    # Advanced pitcher metrics
    Column("fip", Float, server_default=text("0")),
    Column("xfip", Float, server_default=text("0")),
    Column("siera", Float, server_default=text("0")),
    # Statcast
    Column("stuff_plus", Float),
    Column("location_plus", Float),
    Column("pitching_plus", Float),
    Column("gmli", Float),
)

Index("idx_season_stats_player", season_stats.c.player_id)
season_stats.append_constraint(_PKC("player_id", "season"))

# ── 5. ros_projections ───────────────────────────────────────────────────────
ros_projections = Table(
    "ros_projections",
    metadata,
    Column("player_id", Integer, nullable=False),
    Column("system", Text, nullable=False),
    Column("pa", Integer, server_default=text("0")),
    Column("ab", Integer, server_default=text("0")),
    Column("h", Integer, server_default=text("0")),
    Column("r", Integer, server_default=text("0")),
    Column("hr", Integer, server_default=text("0")),
    Column("rbi", Integer, server_default=text("0")),
    Column("sb", Integer, server_default=text("0")),
    Column("avg", Float, server_default=text("0")),
    Column("ip", Float, server_default=text("0")),
    Column("w", Integer, server_default=text("0")),
    Column("sv", Integer, server_default=text("0")),
    Column("k", Integer, server_default=text("0")),
    Column("era", Float, server_default=text("0")),
    Column("whip", Float, server_default=text("0")),
    Column("er", Integer, server_default=text("0")),
    Column("bb_allowed", Integer, server_default=text("0")),
    Column("h_allowed", Integer, server_default=text("0")),
    Column("updated_at", Text),
    # Phase 1
    Column("obp", Float, server_default=text("0")),
    Column("l", Integer, server_default=text("0")),
    Column("bb", Integer, server_default=text("0")),
    Column("hbp", Integer, server_default=text("0")),
    Column("sf", Integer, server_default=text("0")),
    # Advanced pitcher metrics
    Column("fip", Float, server_default=text("0")),
    Column("xfip", Float, server_default=text("0")),
    Column("siera", Float, server_default=text("0")),
    # Statcast
    Column("stuff_plus", Float),
    Column("location_plus", Float),
    Column("pitching_plus", Float),
)

ros_projections.append_constraint(_PKC("player_id", "system"))
Index("idx_ros_proj_player", ros_projections.c.player_id)

# ── 6. league_rosters ────────────────────────────────────────────────────────
league_rosters = Table(
    "league_rosters",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("team_name", Text, nullable=False),
    Column("team_index", Integer, nullable=False),
    Column("player_id", Integer, nullable=False),
    Column("roster_slot", Text),
    Column("is_user_team", Integer, nullable=False, server_default=text("0")),
    # _safe_add_column additions
    Column("status", Text, server_default=text("'active'")),
    Column("selected_position", Text, server_default=text("''")),
    Column("editorial_team_abbr", Text, server_default=text("''")),
    Column("yahoo_player_key", Text, server_default=text("''")),
    Column("is_undroppable", Integer, server_default=text("0")),
    UniqueConstraint("team_name", "player_id", name="uq_league_rosters_team_player"),
)

Index("idx_league_rosters_team", league_rosters.c.team_name)

# ── 7. league_teams ──────────────────────────────────────────────────────────
league_teams = Table(
    "league_teams",
    metadata,
    Column("team_key", Text, primary_key=True),
    Column("team_name", Text, nullable=False),
    Column("team_index", Integer),
    Column("logo_url", Text),
    Column("manager_name", Text),
    Column("is_user_team", Integer, server_default=text("0")),
    # _safe_add_column additions
    Column("faab_balance", Float),
    Column("waiver_priority", Integer),
    Column("number_of_moves", Integer),
    Column("number_of_trades", Integer),
)

# ── 8. league_standings ──────────────────────────────────────────────────────
league_standings = Table(
    "league_standings",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("team_name", Text, nullable=False),
    Column("category", Text, nullable=False),
    Column("total", Float, server_default=text("0")),
    Column("rank", Integer),
    Column("points", Float),
    UniqueConstraint("team_name", "category", name="uq_league_standings_team_cat"),
)

# ── 9. park_factors ──────────────────────────────────────────────────────────
park_factors = Table(
    "park_factors",
    metadata,
    Column("team_code", Text, primary_key=True),
    Column("factor_hitting", Float, server_default=text("1.0")),
    Column("factor_pitching", Float, server_default=text("1.0")),
)

# ── 10. refresh_log ──────────────────────────────────────────────────────────
refresh_log = Table(
    "refresh_log",
    metadata,
    Column("source", Text, primary_key=True),
    Column("last_refresh", Text),
    Column("status", Text, server_default=text("'unknown'")),
    Column("rows_written", Integer),
    Column("rows_expected_min", Integer),
    Column("message", Text),
    Column("tier", Text, server_default=text("'primary'")),
)

# ── 11. injury_history ───────────────────────────────────────────────────────
injury_history = Table(
    "injury_history",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("player_id", Integer, nullable=False),
    Column("season", Integer, nullable=False),
    Column("games_played", Integer, server_default=text("0")),
    Column("games_available", Integer, server_default=text("162")),
    Column("il_stints", Integer, server_default=text("0")),
    Column("il_days", Integer, server_default=text("0")),
    Column("created_at", Text),
)

Index("idx_injury_history_player", injury_history.c.player_id)
Index(
    "idx_injury_history_player_season",
    injury_history.c.player_id,
    injury_history.c.season,
    unique=True,
)

# ── 12. transactions ─────────────────────────────────────────────────────────
transactions = Table(
    "transactions",
    metadata,
    Column("transaction_id", Integer, primary_key=True),
    Column("league_id", Text),
    Column("player_id", Integer, nullable=False),
    Column("type", Text, nullable=False),
    Column("team_from", Text),
    Column("team_to", Text),
    Column("timestamp", Text),
    Column("created_at", Text),
)

Index("idx_transactions_player", transactions.c.player_id)

# ── 13. statcast_archive ─────────────────────────────────────────────────────
statcast_archive = Table(
    "statcast_archive",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("player_id", Integer, nullable=False),
    Column("season", Integer, nullable=False),
    Column("ev_mean", Float),
    Column("ev_p90", Float),
    Column("barrel_pct", Float),
    Column("hard_hit_pct", Float),
    Column("xba", Float),
    Column("xslg", Float),
    Column("xwoba", Float),
    Column("whiff_pct", Float),
    Column("chase_rate", Float),
    Column("sprint_speed", Float),
    Column("ff_avg_speed", Float),
    Column("ff_spin_rate", Float),
    Column("k_pct", Float),
    Column("bb_pct", Float),
    Column("gb_pct", Float),
    Column("stuff_plus", Float),
    Column("location_plus", Float),
    Column("pitching_plus", Float),
    Column("gmli", Float),
    Column("babip", Float),
    Column("iso", Float),
    Column("hitter_k_pct", Float),
    Column("hitter_bb_pct", Float),
    Column("ld_pct", Float),
    Column("hitter_fb_pct", Float),
    Column("hitter_gb_pct", Float),
    Column("bat_speed", Float),
    Column("updated_at", Text, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("player_id", "season", name="uq_statcast_archive_player_season"),
)

Index("idx_statcast_archive_player", statcast_archive.c.player_id)
Index(
    "idx_statcast_archive_player_season",
    statcast_archive.c.player_id,
    statcast_archive.c.season,
    unique=True,
)

# ── 14. player_tags ──────────────────────────────────────────────────────────
player_tags = Table(
    "player_tags",
    metadata,
    Column("player_id", Integer, nullable=False),
    Column(
        "tag",
        Text,
        CheckConstraint("tag IN ('Sleeper','Target','Avoid','Breakout','Bust')", name="ck_player_tags_tag"),
        nullable=False,
    ),
    Column("note", Text, server_default=text("''")),
    Column("created_at", Text, server_default=func.now()),
)

player_tags.append_constraint(_PKC("player_id", "tag"))

# ── 15. leagues ──────────────────────────────────────────────────────────────
leagues = Table(
    "leagues",
    metadata,
    Column("league_id", Text, primary_key=True),
    Column("platform", Text, server_default=text("'manual'")),
    Column("league_name", Text, nullable=False),
    Column("num_teams", Integer, server_default=text("12")),
    Column("scoring_format", Text, server_default=text("'h2h_categories'")),
    Column("yahoo_league_id", Text),
    Column("created_at", Text, nullable=False),
    Column("is_active", Integer, server_default=text("0")),
)

# ── 16. prospect_rankings ────────────────────────────────────────────────────
prospect_rankings = Table(
    "prospect_rankings",
    metadata,
    Column("prospect_id", Integer, primary_key=True),
    Column("mlb_id", Integer),
    Column("name", Text, nullable=False),
    Column("team", Text),
    Column("position", Text),
    Column("fg_rank", Integer),
    Column("fg_fv", Integer),
    Column("fg_eta", Text),
    Column("fg_risk", Text),
    Column("age", Integer),
    Column("hit_present", Integer),
    Column("hit_future", Integer),
    Column("game_present", Integer),
    Column("game_future", Integer),
    Column("raw_present", Integer),
    Column("raw_future", Integer),
    Column("speed", Integer),
    Column("field", Integer),
    Column("ctrl_present", Integer),
    Column("ctrl_future", Integer),
    Column("scouting_report", Text),
    Column("tldr", Text),
    Column("milb_level", Text),
    Column("milb_avg", Float),
    Column("milb_obp", Float),
    Column("milb_slg", Float),
    Column("milb_k_pct", Float),
    Column("milb_bb_pct", Float),
    Column("milb_hr", Integer),
    Column("milb_sb", Integer),
    Column("milb_ip", Float),
    Column("milb_era", Float),
    Column("milb_whip", Float),
    Column("milb_k9", Float),
    Column("milb_bb9", Float),
    Column("readiness_score", Float),
    Column("fetched_at", Text),
)

# ── 17. ecr_consensus ────────────────────────────────────────────────────────
ecr_consensus = Table(
    "ecr_consensus",
    metadata,
    Column("player_id", Integer, primary_key=True),
    Column("espn_rank", Integer),
    Column("yahoo_adp", Float),
    Column("cbs_rank", Integer),
    Column("nfbc_adp", Float),
    Column("fg_adp", Float),
    Column("fp_ecr", Integer),
    Column("heater_sgp_rank", Integer),
    Column("consensus_rank", Integer),
    Column("consensus_avg", Float),
    Column("rank_min", Integer),
    Column("rank_max", Integer),
    Column("rank_stddev", Float),
    Column("n_sources", Integer),
    Column("fetched_at", Text),
)

# ── 18. player_id_map ────────────────────────────────────────────────────────
player_id_map = Table(
    "player_id_map",
    metadata,
    Column("player_id", Integer, primary_key=True),
    Column("espn_id", Integer),
    Column("yahoo_key", Text),
    Column("fg_id", Integer),
    Column("mlb_id", Integer),
    Column("cbs_id", Integer),
    Column("nfbc_id", Integer),
    Column("name", Text),
    Column("team", Text),
    Column("updated_at", Text),
)

# Partial unique indexes — only where the column IS NOT NULL
Index(
    "idx_id_map_espn",
    player_id_map.c.espn_id,
    unique=True,
    postgresql_where=text("espn_id IS NOT NULL"),
    sqlite_where=text("espn_id IS NOT NULL"),
)
Index(
    "idx_id_map_yahoo",
    player_id_map.c.yahoo_key,
    unique=True,
    postgresql_where=text("yahoo_key IS NOT NULL"),
    sqlite_where=text("yahoo_key IS NOT NULL"),
)
Index(
    "idx_id_map_fg",
    player_id_map.c.fg_id,
    unique=True,
    postgresql_where=text("fg_id IS NOT NULL"),
    sqlite_where=text("fg_id IS NOT NULL"),
)
Index(
    "idx_id_map_mlb",
    player_id_map.c.mlb_id,
    unique=True,
    postgresql_where=text("mlb_id IS NOT NULL"),
    sqlite_where=text("mlb_id IS NOT NULL"),
)

# ── 19. player_news ──────────────────────────────────────────────────────────
player_news = Table(
    "player_news",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("player_id", Integer, nullable=False),
    Column("source", Text, nullable=False),
    Column("headline", Text, nullable=False),
    Column("detail", Text),
    Column("news_type", Text),
    Column("injury_body_part", Text),
    Column("il_status", Text),
    Column("sentiment_score", Float),
    Column("published_at", Text),
    Column("fetched_at", Text),
    UniqueConstraint("player_id", "source", "headline", "published_at", name="uq_player_news"),
)

Index("idx_player_news_player", player_news.c.player_id)
Index("idx_player_news_type", player_news.c.news_type)

# ── 20. ownership_trends ─────────────────────────────────────────────────────
ownership_trends = Table(
    "ownership_trends",
    metadata,
    Column("player_id", Integer, nullable=False),
    Column("date", Text, nullable=False),
    Column("percent_owned", Float),
    Column("delta_7d", Float),
)

ownership_trends.append_constraint(_PKC("player_id", "date"))

# ── 21. opponent_profiles ────────────────────────────────────────────────────
opponent_profiles = Table(
    "opponent_profiles",
    metadata,
    Column("team_name", Text, primary_key=True),
    Column("tier", Integer, nullable=False, server_default=text("3")),
    Column("threat_level", Text),
    Column("strengths", Text),
    Column("weaknesses", Text),
    Column("manager", Text),
    Column("notes", Text),
)

# ── 22. league_schedule ──────────────────────────────────────────────────────
league_schedule = Table(
    "league_schedule",
    metadata,
    Column("week", Integer, nullable=False),
    Column("team_a", Text, nullable=False),
    Column("team_b", Text, nullable=False),
)

league_schedule.append_constraint(_PKC("week", "team_a"))

# ── 23. league_schedule_full ─────────────────────────────────────────────────
league_schedule_full = Table(
    "league_schedule_full",
    metadata,
    Column("week", Integer, nullable=False),
    Column("team_a", Text, nullable=False),
    Column("team_b", Text, nullable=False),
)

league_schedule_full.append_constraint(_PKC("week", "team_a", "team_b"))

# ── 24. league_records ───────────────────────────────────────────────────────
league_records = Table(
    "league_records",
    metadata,
    Column("team_name", Text, primary_key=True),
    Column("wins", Integer, server_default=text("0")),
    Column("losses", Integer, server_default=text("0")),
    Column("ties", Integer, server_default=text("0")),
    Column("win_pct", Float, server_default=text("0.0")),
    Column("points_for", Float, server_default=text("0.0")),
    Column("points_against", Float, server_default=text("0.0")),
    Column("streak", Text, server_default=text("''")),
    Column("rank", Integer, server_default=text("0")),
    Column("updated_at", Text),
)

# ── 25. league_matchup_cache ─────────────────────────────────────────────────
league_matchup_cache = Table(
    "league_matchup_cache",
    metadata,
    Column("team_name", Text, nullable=False),
    Column("week", Integer, nullable=False),
    Column("opp_name", Text),
    Column("matchup_json", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)

league_matchup_cache.append_constraint(_PKC("team_name", "week"))

# ── 26. league_settings ──────────────────────────────────────────────────────
league_settings = Table(
    "league_settings",
    metadata,
    Column(
        "id",
        Integer,
        CheckConstraint("id = 1", name="ck_league_settings_single_row"),
        primary_key=True,
        autoincrement=False,  # single-row table (CHECK id=1) — INTEGER PK on PG, not SERIAL
    ),
    Column("settings_json", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)

# ── 27. yahoo_free_agents ────────────────────────────────────────────────────
yahoo_free_agents = Table(
    "yahoo_free_agents",
    metadata,
    Column("player_key", Text, primary_key=True),
    Column("player_name", Text, nullable=False),
    Column("positions", Text),
    Column("team", Text),
    Column("percent_owned", Float),
    Column("fetched_at", Text),
)

# ── 28. roster_snapshots ─────────────────────────────────────────────────────
roster_snapshots = Table(
    "roster_snapshots",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("snapshot_date", Text, nullable=False),
    Column("team_name", Text, nullable=False),
    Column("player_id", Integer, nullable=False),
    Column("roster_slot", Text),
    UniqueConstraint("snapshot_date", "team_name", "player_id", name="uq_roster_snapshots"),
)

# ── 29. game_day_weather ─────────────────────────────────────────────────────
game_day_weather = Table(
    "game_day_weather",
    metadata,
    Column("game_pk", Integer, primary_key=True),
    Column("game_date", Text, nullable=False),
    Column("venue_team", Text, nullable=False),
    Column("temp_f", Float),
    Column("wind_mph", Float),
    Column("wind_dir", Text),
    Column("precip_pct", Float),
    Column("humidity_pct", Float),
    Column("fetched_at", Text, nullable=False),
)

# ── 30. team_strength ────────────────────────────────────────────────────────
team_strength = Table(
    "team_strength",
    metadata,
    Column("team_abbr", Text, nullable=False),
    Column("season", Integer, nullable=False),
    Column("wrc_plus", Float),
    Column("fip", Float),
    Column("team_ops", Float),
    Column("team_era", Float),
    Column("team_whip", Float),
    Column("k_pct", Float),
    Column("bb_pct", Float),
    Column("fetched_at", Text, nullable=False),
)

team_strength.append_constraint(_PKC("team_abbr", "season"))

# ── 31. opp_pitcher_stats ────────────────────────────────────────────────────
opp_pitcher_stats = Table(
    "opp_pitcher_stats",
    metadata,
    Column("pitcher_id", Integer, nullable=False),
    Column("season", Integer, nullable=False),
    Column("name", Text),
    Column("team", Text),
    Column("era", Float),
    Column("fip", Float),
    Column("xfip", Float),
    Column("whip", Float),
    Column("k_per_9", Float),
    Column("bb_per_9", Float),
    Column("vs_lhb_avg", Float),
    Column("vs_rhb_avg", Float),
    Column("ip", Float),
    Column("hand", Text),
    Column("fetched_at", Text, nullable=False),
)

opp_pitcher_stats.append_constraint(_PKC("pitcher_id", "season"))

# ── 32. umpire_tendencies ────────────────────────────────────────────────────
umpire_tendencies = Table(
    "umpire_tendencies",
    metadata,
    Column("umpire_name", Text, primary_key=True),
    Column("games_umped", Integer, server_default=text("0")),
    Column("k_pct", Float, server_default=text("0.0")),
    Column("bb_pct", Float, server_default=text("0.0")),
    Column("runs_per_game", Float, server_default=text("0.0")),
    Column("k_pct_delta", Float, server_default=text("0.0")),
    Column("bb_pct_delta", Float, server_default=text("0.0")),
    Column("run_env_delta", Float, server_default=text("0.0")),
    Column("season", Integer, nullable=False),
    Column("fetched_at", Text, nullable=False),
)

# ── 33. catcher_framing ──────────────────────────────────────────────────────
catcher_framing = Table(
    "catcher_framing",
    metadata,
    Column("player_id", Integer, nullable=False),
    Column("season", Integer, nullable=False),
    Column("framing_runs", Float, server_default=text("0.0")),
    Column("framing_runs_per_game", Float, server_default=text("0.0")),
    Column("pop_time", Float, server_default=text("0.0")),
    Column("cs_pct", Float, server_default=text("0.0")),
    Column("games_caught", Integer, server_default=text("0")),
    Column("fetched_at", Text, nullable=False),
)

catcher_framing.append_constraint(_PKC("player_id", "season"))

# ── 34. pvb_splits ───────────────────────────────────────────────────────────
pvb_splits = Table(
    "pvb_splits",
    metadata,
    Column("batter_id", Integer, nullable=False),
    Column("pitcher_id", Integer, nullable=False),
    Column("pa", Integer, server_default=text("0")),
    Column("avg", Float, server_default=text("0.0")),
    Column("obp", Float, server_default=text("0.0")),
    Column("slg", Float, server_default=text("0.0")),
    Column("hr", Integer, server_default=text("0")),
    Column("k", Integer, server_default=text("0")),
    Column("bb", Integer, server_default=text("0")),
    Column("woba", Float, server_default=text("0.0")),
    Column("fetched_at", Text, nullable=False),
)

pvb_splits.append_constraint(_PKC("batter_id", "pitcher_id"))
Index("idx_pvb_batter", pvb_splits.c.batter_id)
Index("idx_pvb_pitcher", pvb_splits.c.pitcher_id)

# ── 35. opponent_trade_history ───────────────────────────────────────────────
opponent_trade_history = Table(
    "opponent_trade_history",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("opponent_team", Text, nullable=False),
    Column("trade_date", Text, nullable=False),
    Column("gave_player_ids", Text),
    Column("received_player_ids", Text),
    Column("gave_sgp", Float, server_default=text("0.0")),
    Column("received_sgp", Float, server_default=text("0.0")),
    Column("category_focus", Text),
    Column("position_focus", Text),
    Column("accepted", Integer, server_default=text("1")),
    Column("season", Integer, nullable=False),
)

Index("idx_opp_trade_team", opponent_trade_history.c.opponent_team)

# ── 36. game_logs ────────────────────────────────────────────────────────────
game_logs = Table(
    "game_logs",
    metadata,
    Column("player_id", Integer, nullable=False),
    Column("game_date", Text, nullable=False),
    Column("season", Integer, nullable=False, server_default=text("2026")),
    Column("pa", Integer, server_default=text("0")),
    Column("ab", Integer, server_default=text("0")),
    Column("h", Integer, server_default=text("0")),
    Column("r", Integer, server_default=text("0")),
    Column("hr", Integer, server_default=text("0")),
    Column("rbi", Integer, server_default=text("0")),
    Column("sb", Integer, server_default=text("0")),
    Column("bb", Integer, server_default=text("0")),
    Column("hbp", Integer, server_default=text("0")),
    Column("sf", Integer, server_default=text("0")),
    Column("ip", Float, server_default=text("0.0")),
    Column("w", Integer, server_default=text("0")),
    Column("l", Integer, server_default=text("0")),
    Column("sv", Integer, server_default=text("0")),
    Column("k", Integer, server_default=text("0")),
    Column("er", Integer, server_default=text("0")),
    Column("bb_allowed", Integer, server_default=text("0")),
    Column("h_allowed", Integer, server_default=text("0")),
    # Combustion dossier columns (_safe_add_column already in CREATE TABLE definition)
    Column("opponent_id", Integer),
    Column("opponent_abbr", Text),
    Column("is_home", Integer),
    Column("result", Text),
    Column("team_score", Integer),
    Column("opp_score", Integer),
)

game_logs.append_constraint(_PKC("player_id", "game_date"))

# ── 37. league_draft_picks ───────────────────────────────────────────────────
league_draft_picks = Table(
    "league_draft_picks",
    metadata,
    Column("pick_number", Integer, primary_key=True),
    Column("round", Integer, nullable=False),
    Column("team_name", Text, nullable=False),
    Column("player_id", Integer),
    Column("player_name", Text, nullable=False),
)

# ── 38. users ────────────────────────────────────────────────────────────────
# username uses COLLATE NOCASE on SQLite -> CITEXT on Postgres
users = Table(
    "users",
    metadata,
    Column("user_id", Integer, primary_key=True),
    Column(
        "username",
        String().with_variant(postgresql.CITEXT(), "postgresql"),
        nullable=False,
    ),
    Column("password_hash", Text, nullable=False),
    Column("display_name", Text),
    Column("team_name", Text),
    Column("status", Text, nullable=False, server_default=text("'pending'")),
    Column("is_admin", Integer, nullable=False, server_default=text("0")),
    Column("created_at", Text, nullable=False),
    Column("approved_at", Text),
    Column("approved_by", Text),
    Column("last_seen_at", Text),
    UniqueConstraint("username", name="uq_users_username"),
)

Index("idx_users_status", users.c.status)

# ── 39. feedback ─────────────────────────────────────────────────────────────
feedback = Table(
    "feedback",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("page", Text, nullable=False),
    Column("feature_tag", Text),
    Column("message", Text, nullable=False),
    Column("app_version", Text, nullable=False),
    Column("data_state", Text),
    Column("status", Text, nullable=False, server_default=text("'new'")),
    Column("admin_notes", Text),
    Column("created_at", Text, nullable=False),
)

Index("idx_feedback_status", feedback.c.status)

# ── 40. usage_events ─────────────────────────────────────────────────────────
usage_events = Table(
    "usage_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("page", Text, nullable=False),
    Column("action", Text, nullable=False, server_default=text("'view'")),
    Column("session_id", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)

Index("idx_usage_user_created", usage_events.c.user_id, usage_events.c.created_at)

# ── 41. feature_flags ────────────────────────────────────────────────────────
feature_flags = Table(
    "feature_flags",
    metadata,
    Column("key", Text, primary_key=True),
    Column("enabled", Integer, nullable=False, server_default=text("1")),
    Column("updated_by", Integer),
    Column("updated_at", Text),
)

# ── 42. audit_log ────────────────────────────────────────────────────────────
audit_log = Table(
    "audit_log",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("admin_id", Integer, nullable=False),
    Column("action", Text, nullable=False),
    Column("target", Text),
    Column("detail", Text),
    Column("created_at", Text, nullable=False),
)

Index("idx_audit_created", audit_log.c.created_at)

# ── 43. app_settings ─────────────────────────────────────────────────────────
app_settings = Table(
    "app_settings",
    metadata,
    Column("key", Text, primary_key=True),
    Column("value", Text),
    Column("updated_by", Integer),
    Column("updated_at", Text),
)

# ── 44. sessions ─────────────────────────────────────────────────────────────
sessions = Table(
    "sessions",
    metadata,
    Column("session_id", Text, primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("login_at", Text, nullable=False),
    Column("last_activity_at", Text, nullable=False),
)

Index("idx_sessions_user", sessions.c.user_id)

# ── 45. page_visits ──────────────────────────────────────────────────────────
page_visits = Table(
    "page_visits",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", Text, nullable=False),
    Column("user_id", Integer, nullable=False),
    Column("page", Text, nullable=False),
    Column("enter_at", Text, nullable=False),
    Column("exit_at", Text),
    Column("dwell_seconds", Float),
)

Index("idx_page_visits_session", page_visits.c.session_id)

# ── 46. auth_tokens ──────────────────────────────────────────────────────────
auth_tokens = Table(
    "auth_tokens",
    metadata,
    Column("token", Text, primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("created_at", Text, nullable=False),
    Column("expires_at", Text, nullable=False),
    Column("revoked", Integer, nullable=False, server_default=text("0")),
)

Index("idx_auth_tokens_user", auth_tokens.c.user_id)

# ── 47. ai_provider_keys ─────────────────────────────────────────────────────
ai_provider_keys = Table(
    "ai_provider_keys",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("provider", Text, nullable=False),
    Column("label", Text),
    Column("encrypted_key", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    UniqueConstraint("user_id", "provider", "label", name="uq_ai_provider_keys"),
)

Index("idx_ai_keys_user", ai_provider_keys.c.user_id)

# ── 48. ai_conversations ─────────────────────────────────────────────────────
ai_conversations = Table(
    "ai_conversations",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("title", Text, nullable=False),
    Column("model", Text),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)

Index("idx_ai_conv_user", ai_conversations.c.user_id, ai_conversations.c.updated_at)

# ── 49. ai_messages ──────────────────────────────────────────────────────────
ai_messages = Table(
    "ai_messages",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("conversation_id", Integer, nullable=False),
    Column("role", Text, nullable=False),
    Column("content", Text, nullable=False),
    Column("model", Text),
    Column("tokens_in", Integer, server_default=text("0")),
    Column("tokens_out", Integer, server_default=text("0")),
    Column("cost_usd", Float, server_default=text("0.0")),
    Column("created_at", Text, nullable=False),
)

Index("idx_ai_msg_conv", ai_messages.c.conversation_id, ai_messages.c.id)

# ── 50. ai_usage_ledger ──────────────────────────────────────────────────────
ai_usage_ledger = Table(
    "ai_usage_ledger",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, nullable=False),
    Column("day", Text, nullable=False),
    Column("tokens_in", Integer, server_default=text("0")),
    Column("tokens_out", Integer, server_default=text("0")),
    Column("cost_usd", Float, server_default=text("0.0")),
    UniqueConstraint("user_id", "day", name="uq_ai_usage_ledger"),
)

# ── 51. forced_refresh_queue ─────────────────────────────────────────────────
forced_refresh_queue = Table(
    "forced_refresh_queue",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("source", Text, nullable=False),
    Column("requested_by", Integer),
    Column("status", Text, nullable=False, server_default=text("'pending'")),
    Column("detail", Text),
    Column("created_at", Text, nullable=False),
    Column("completed_at", Text),
)

Index("idx_refresh_queue_status", forced_refresh_queue.c.status)

# ── 52. user_watchlist ───────────────────────────────────────────────────────
user_watchlist = Table(
    "user_watchlist",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, nullable=False, server_default=text("0")),
    Column("player_id", Integer, nullable=False),
    Column("created_at", Text, nullable=False),
    UniqueConstraint("user_id", "player_id", name="uq_user_watchlist"),
)

Index("idx_watchlist_user", user_watchlist.c.user_id)

# ── 53. user_saved_views ─────────────────────────────────────────────────────
user_saved_views = Table(
    "user_saved_views",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, nullable=False, server_default=text("0")),
    Column("kind", Text, nullable=False),
    Column("name", Text, nullable=False),
    Column("payload_json", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    UniqueConstraint("user_id", "kind", "name", name="uq_user_saved_views"),
)

Index("idx_saved_views_user_kind", user_saved_views.c.user_id, user_saved_views.c.kind)
