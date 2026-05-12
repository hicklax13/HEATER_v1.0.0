"""INFRA-F6 fix: bootstrap phases must use update_refresh_log_auto (with row count)
for success states, not plain update_refresh_log(..., "success")."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "src" / "data_bootstrap.py"

# These specific call sites are allowed to use the plain form (status-only updates,
# not "success" claims). Add to this list ONLY with justification.
ALLOWED_PLAIN_SITES = {
    # phase_name: justification
    "draft_results": "early-return path when no Yahoo client; not a success claim",
}


def test_success_calls_use_update_refresh_log_auto():
    """`update_refresh_log("X", "success")` patterns should be migrated to
    `update_refresh_log_auto(...)` so silent 0-row writes are caught."""
    assert BOOTSTRAP.exists()
    text = BOOTSTRAP.read_text(encoding="utf-8")
    # Match both positional `update_refresh_log("x", "success", ...)` and keyword
    # `update_refresh_log("x", status="success", ...)` forms. Either pattern
    # claims success without verifying row count.
    pat = re.compile(
        r'update_refresh_log\(\s*"([a-z_]+)"\s*,\s*(?:status\s*=\s*)?"success"',
        re.MULTILINE,
    )
    offenders: list[tuple[int, str, str]] = []
    for m in pat.finditer(text):
        phase = m.group(1)
        if phase in ALLOWED_PLAIN_SITES:
            continue
        lineno = text[: m.start()].count("\n") + 1
        offenders.append((lineno, phase, m.group(0)))
    assert not offenders, (
        f"INFRA-F6 regression: bootstrap phase(s) use plain "
        f"update_refresh_log(..., 'success') without row-count gate. "
        f"Migrate to update_refresh_log_auto(name, count, expected_min, message=...). "
        f"Offenders: {offenders}"
    )
