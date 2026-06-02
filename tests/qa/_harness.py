"""Run any HEATER page headlessly as a chosen team/role and report what broke.

Usage from other qa tests — PREFER the conftest fixture (no import boilerplate,
and it sidesteps the installed `tests` package collision in .venv):

    def test_my_page(run_page_as_team, team_names):
        r = run_page_as_team("pages/1_My_Team.py", team_names[0])
        assert r.ran and r.exception is None and not r.errors

(The fixture lives in tests/qa/conftest.py and simply returns this function.)
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
    # Rendered output (populated only on a successful run) so deep per-page
    # tests can assert value-plausibility, not just crash-freedom.
    dataframes: list = field(default_factory=list)  # every st.dataframe / st.table value
    metrics: list = field(default_factory=list)  # [{"label","value"}] from st.metric
    markdown: str = ""  # concatenated st.markdown text (custom HTML tables live here)
    # st.subheader/header/title/caption/text are SEPARATE element types and are NOT
    # part of `markdown`. Captured here so heading-presence checks work.
    headings: list[str] = field(default_factory=list)
    # multiselect / selectbox option labels (option text is NOT in markdown).
    widget_options: list[str] = field(default_factory=list)
    # Unified text corpus (markdown + headings + metric pairs + dataframe text) so
    # value-plausibility scans catch a value regardless of which channel rendered it.
    text: str = ""


# ── Core harness ─────────────────────────────────────────────────────────────


def run_page_as_team(
    page_path: str,
    team_name: str,
    is_admin: bool = False,
    timeout: float = 180.0,
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
        AppTest run timeout in seconds (default 180s). Under the test network
        guard, each Yahoo fetch burns ~15s (backoff retries until the 15s
        ThreadPoolExecutor cap) before falling back to SQLite, so a page that
        makes several Yahoo calls (e.g. Closer Monitor) can take 60-120s to
        render — well above live-Yahoo latency. 180s absorbs that plus machine
        load variance; a genuinely hung render is still bounded.

    Returns
    -------
    HarnessResult
        ``.ran`` True if ``at.run()`` completed without raising a Python
        exception.  ``.exception`` is the page-level exception string (from
        ``st.exception`` elements) or the Python exception message.  ``.errors``
        lists all ``st.error`` message bodies.
    """
    # Scope MULTI_USER to this call only, restoring it in the finally below.
    # Without this, the harness would leave MULTI_USER=1 set process-wide and
    # leak into MULTI_USER-off (v1 back-compat) tests when the qa suite shares a
    # process with the main suite (e.g. the Task-6 full-suite re-verification),
    # silently corrupting their results.
    _prev_multi_user = os.environ.get("MULTI_USER")
    os.environ["MULTI_USER"] = "1"
    try:
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
                exception=(
                    f"QA user {username!r} not found — run `.venv\\Scripts\\python.exe scripts\\qa_seed_local.py`"
                ),
            )

        # Reset module-global league caches so one team's totals / FA pool can't
        # bleed into the next team's render in this shared serial process. The
        # standings_utils caches hold league-global data (so this is defense-in-
        # depth, not a production bug), but it keeps per-team value/ownership
        # assertions honest. See the 2026-06-02 silent-failure audit, Finding 2.
        try:
            from src import standings_utils

            standings_utils.clear_cache()
        except Exception:  # noqa: BLE001
            pass

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

        # Capture rendered data so deep per-page tests can assert value
        # plausibility (not just crash-freedom). Index-robust: collect ALL
        # dataframes/tables/metrics + the concatenated markdown text.
        dataframes: list = []
        for elem in list(getattr(at, "dataframe", []) or []):
            try:
                dataframes.append(elem.value)
            except Exception:  # noqa: BLE001
                pass
        try:
            for elem in list(getattr(at, "table", []) or []):
                try:
                    dataframes.append(elem.value)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001 - some Streamlit versions lack .table
            pass

        metrics: list = []
        for elem in list(getattr(at, "metric", []) or []):
            try:
                metrics.append({"label": elem.label, "value": elem.value})
            except Exception:  # noqa: BLE001
                pass

        markdown_text = "\n".join(getattr(elem, "value", "") or "" for elem in list(getattr(at, "markdown", []) or []))

        # Headings / captions / plain text — these are SEPARATE Streamlit element
        # types from st.markdown and are NOT included in markdown_text above.
        headings: list[str] = []
        for _attr in ("title", "header", "subheader", "caption", "text"):
            for elem in list(getattr(at, _attr, []) or []):
                try:
                    val = getattr(elem, "value", None)
                    if val:
                        headings.append(str(val))
                except Exception:  # noqa: BLE001
                    pass

        # Widget option labels (multiselect / selectbox) — option text is NOT in markdown.
        widget_options: list[str] = []
        for _attr in ("multiselect", "selectbox"):
            for elem in list(getattr(at, _attr, []) or []):
                try:
                    opts = getattr(elem, "options", None) or []
                    widget_options.extend(str(o) for o in opts)
                except Exception:  # noqa: BLE001
                    pass

        # Unified text corpus so value-plausibility scans catch a value regardless of
        # which channel rendered it (markdown vs heading vs metric vs dataframe text).
        _text_parts: list[str] = [markdown_text] if markdown_text else []
        _text_parts.extend(headings)
        for _m in metrics:
            _text_parts.append(f"{_m.get('label', '')}: {_m.get('value', '')}")
        for _df in dataframes:
            try:
                _text_parts.append(_df.to_string())
            except Exception:  # noqa: BLE001
                pass
        text_corpus = "\n".join(p for p in _text_parts if p)

        return HarnessResult(
            page=page_path,
            team=team_name,
            is_admin=is_admin,
            ran=True,
            exception=page_exception,
            errors=errors,
            warnings=warnings,
            dataframes=dataframes,
            metrics=metrics,
            markdown=markdown_text,
            headings=headings,
            widget_options=widget_options,
            text=text_corpus,
        )
    finally:
        # Restore prior MULTI_USER (None → unset) so there is no env bleed.
        if _prev_multi_user is None:
            os.environ.pop("MULTI_USER", None)
        else:
            os.environ["MULTI_USER"] = _prev_multi_user
