"""Plan 4 guard: the three Railway deploy artifacts exist and carry the
load-bearing invariants (3.12-slim base, --no-deps yfpy/streamlit-oauth,
headless $PORT bind, single replica, healthcheck path, seed-dir preserved)."""

from pathlib import Path

_ROOT = Path(__file__).parent.parent
_DOCKERFILE = _ROOT / "Dockerfile"
_RAILWAY = _ROOT / "railway.toml"
_DOCKERIGNORE = _ROOT / ".dockerignore"


def test_dockerfile_exists_and_uses_312_slim():
    assert _DOCKERFILE.exists(), "Dockerfile missing"
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert "python:3.12-slim" in text, "must pin python:3.12-slim (CI parity)"


def test_dockerfile_installs_no_deps_yahoo_wrappers():
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert "--no-deps" in text, "yfpy/streamlit-oauth need --no-deps (dotenv pin conflict)"
    assert "yfpy" in text
    assert "streamlit-oauth" in text


def test_dockerfile_binds_port_headless_all_interfaces():
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert "PORT" in text, "must bind the platform-injected $PORT"
    assert "--server.headless" in text
    assert "0.0.0.0" in text, "must bind all interfaces inside the container"


def test_dockerignore_excludes_heavy_and_secret_paths():
    text = _DOCKERIGNORE.read_text(encoding="utf-8")
    lines = {ln.strip() for ln in text.splitlines()}
    for needle in (".venv", ".git", "__pycache__", ".claude/"):
        assert needle in text, f".dockerignore must exclude {needle}"
    # A bare-filename block-list (data/draft_tool.db) silently leaks sibling files
    # like data/draft_tool.db.backup-*. The data/* allow-list excludes the runtime
    # DB, WAL/SHM, backups, and token in one shot.
    assert "data/*" in lines, ".dockerignore must use the data/* allow-list, not bare-file excludes"


def test_dockerignore_keeps_seed_dir():
    # Tier 3 fallbacks (catcher_framing, umpire_tendencies, park_factors) ship in
    # data/seed/ and MUST be in the image. A bare `data/` exclude would break them;
    # the data/* + !data/seed/ allow-list excludes the rest while re-including seed.
    text = _DOCKERIGNORE.read_text(encoding="utf-8")
    lines = {ln.strip() for ln in text.splitlines()}
    assert "data/" not in lines, "must not blanket-exclude data/ (would drop data/seed/)"
    assert "data/seed/" not in lines, "data/seed/ must remain in the image"
    assert "!data/seed/" in lines, "data/seed/ must be re-included after the data/* exclude"


def test_railway_uses_dockerfile_builder():
    assert _RAILWAY.exists(), "railway.toml missing"
    text = _RAILWAY.read_text(encoding="utf-8")
    assert 'builder = "DOCKERFILE"' in text


def test_railway_has_healthcheck_path():
    text = _RAILWAY.read_text(encoding="utf-8")
    assert "/_stcore/health" in text, "Streamlit healthcheck path"


def test_railway_single_replica():
    text = _RAILWAY.read_text(encoding="utf-8")
    assert "numReplicas = 1" in text, "single-writer + SQLite require exactly one replica"
