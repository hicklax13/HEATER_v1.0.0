"""Run any HEATER page headlessly as a chosen team/role and report what broke.

Usage from other qa tests:
    # Load by file path (avoids the installed `tests` package collision):
    import importlib.util, sys
    from pathlib import Path
    _h = Path(__file__).resolve().parent / "_harness.py"
    _spec = importlib.util.spec_from_file_location("qa_harness", _h)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    run_page_as_team = _mod.run_page_as_team
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Ensure repo root is on sys.path ─────────────────────────────────────────
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ── Load qa_seed_local by file path (avoids installed `tests` pkg collision) ─
_seeder_path = _repo_root / "scripts" / "qa_seed_local.py"
_seeder_spec = importlib.util.spec_from_file_location("qa_seed_local", _seeder_path)
_seeder_mod = importlib.util.module_from_spec(_seeder_spec)  # type: ignore[arg-type]
if "qa_seed_local" not in sys.modules:
    sys.modules["qa_seed_local"] = _seeder_mod
    _seeder_spec.loader.exec_module(_seeder_mod)  # type: ignore[union-attr]
else:
    _seeder_mod = sys.modules["qa_seed_local"]

qa_username = _seeder_mod.qa_username

# ── StreamlitAppTest import ──────────────────────────────────────────────────
from streamlit.testing.v1 import AppTest  # noqa: E402

# ── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class HarnessResult:
    page: str
    team: str
    is_admin: bool
    ran: bool
    exception: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Core harness ─────────────────────────────────────────────────────────────


def run_page_as_team(
    page_path: str,
    team_name: str,
    is_admin: bool = False,
    timeout: float = 90.0,
) -> HarnessResult:
    """Run a HEATER page headlessly as a specific team/role.

    Parameters
    ----------
    page_path:
        Repo-root-relative path to the page, e.g. ``"pages/1_My_Team.py"``.
    team_name:
        Exact team name from ``load_league_rosters()["team_name"]``, or the
        special sentinel ``"admin"`` to run as qa_admin (is_admin must be True).
    is_admin:
        When True the ``qa_admin`` user is used instead of the per-team slug.
    timeout:
        AppTest run timeout in seconds (default 90s — pages hit SQLite fallback
        paths under the network guard, which is slower than live Yahoo).

    Returns
    -------
    HarnessResult
        ``.ran`` True if ``at.run()`` completed without raising a Python
        exception.  ``.exception`` is the page-level exception string (from
        ``st.exception`` elements) or the Python exception message.  ``.errors``
        lists all ``st.error`` message bodies.
    """
    # Must be set BEFORE AppTest runs so multi_user_enabled() returns True
    # when the page module is imported.
    os.environ["MULTI_USER"] = "1"

    username = "qa_admin" if is_admin else qa_username(team_name)

    # Late import so the MULTI_USER env is already set when auth.py loads.
    from src.auth import get_user

    user = get_user(username)
    if user is None:
        return HarnessResult(
            page_path,
            team_name,
            is_admin,
            ran=False,
            exception=(f"QA user {username!r} not found — run `.venv\\Scripts\\python.exe scripts\\qa_seed_local.py`"),
        )

    # Build absolute path relative to repo root for AppTest.
    abs_path = str(_repo_root / page_path)

    at = AppTest.from_file(abs_path, default_timeout=timeout)

    # Seed the session so require_auth() passes without a login redirect.
    # _auth_bootstrap_done skips _ensure_session_bootstrap() (init_db already ran).
    at.session_state["auth_user"] = dict(user)
    at.session_state["_auth_bootstrap_done"] = True

    try:
        at.run()
    except Exception as e:  # page raised before/while rendering
        return HarnessResult(
            page=page_path,
            team=team_name,
            is_admin=is_admin,
            ran=False,
            exception=f"{type(e).__name__}: {e}",
        )

    # Extract exception elements (st.exception() calls within the page).
    page_exception: str | None = None
    if at.exception:
        try:
            page_exception = at.exception[0].value
        except (AttributeError, IndexError):
            page_exception = repr(at.exception[0])

    # Extract st.error() elements.
    errors: list[str] = []
    for elem in at.error:
        try:
            errors.append(elem.value)
        except AttributeError:
            errors.append(repr(elem))

    # Extract st.warning() elements.
    warnings: list[str] = []
    for elem in at.warning:
        try:
            warnings.append(elem.value)
        except AttributeError:
            warnings.append(repr(elem))

    return HarnessResult(
        page=page_path,
        team=team_name,
        is_admin=is_admin,
        ran=True,
        exception=page_exception,
        errors=errors,
        warnings=warnings,
    )
