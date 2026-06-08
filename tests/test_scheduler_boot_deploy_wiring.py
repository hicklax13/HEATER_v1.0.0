"""#9 (2026-06-07): deploy artifacts wire the boot scheduler.

``start.sh`` launches a DEDICATED scheduler process (gated on MULTI_USER) so data
warms at container boot — without waiting for the first browser session — then
execs Streamlit. Both the Dockerfile CMD and the Railway startCommand invoke
start.sh, and start.sh carries the port/headless/bind-all launch invariants.
"""

from pathlib import Path

_ROOT = Path(__file__).parent.parent
_STARTSH = _ROOT / "start.sh"
_DOCKERFILE = _ROOT / "Dockerfile"
_RAILWAY = _ROOT / "railway.toml"


def test_start_sh_exists():
    assert _STARTSH.exists(), "start.sh missing"


def test_start_sh_launches_dedicated_scheduler_gated_on_multi_user():
    text = _STARTSH.read_text(encoding="utf-8")
    assert "HEATER_SCHEDULER_BOOT" in text, "must flag the container boot-managed (sessions read-only)"
    assert "python -m src.scheduler" in text, "must launch the dedicated scheduler process"
    assert "MULTI_USER" in text, "the dedicated writer must be gated on MULTI_USER (v1/local parity)"


def test_start_sh_execs_streamlit_with_bind_invariants():
    text = _STARTSH.read_text(encoding="utf-8")
    assert "streamlit run app.py" in text
    assert "PORT" in text, "must bind the platform-injected $PORT"
    assert "--server.headless" in text
    assert "0.0.0.0" in text, "must bind all interfaces inside the container"


def test_dockerfile_invokes_start_sh():
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert "start.sh" in text, "Dockerfile CMD must invoke start.sh (local parity with Railway)"


def test_railway_start_command_invokes_start_sh():
    text = _RAILWAY.read_text(encoding="utf-8")
    assert "start.sh" in text, "railway.toml startCommand must invoke start.sh"
