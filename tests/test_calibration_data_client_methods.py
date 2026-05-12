"""BUG-016 fix: calibration_data + calibrate_constants call real YahooFantasyClient methods."""

import inspect
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_yahoo_client_methods_used_by_calibration_actually_exist():
    """Every method that calibration_data.py + calibrate_constants.py invokes
    on yahoo_client must exist on YahooFantasyClient."""
    from src.yahoo_api import YahooFantasyClient

    # Get all public methods on YahooFantasyClient
    members = inspect.getmembers(YahooFantasyClient, predicate=inspect.isfunction)
    client_methods = {name for name, _ in members if not name.startswith("_")}
    # Properties (like is_authenticated) are not functions — collect them too
    for name in dir(YahooFantasyClient):
        if name.startswith("_"):
            continue
        attr = getattr(YahooFantasyClient, name, None)
        if isinstance(attr, property):
            client_methods.add(name)

    # Scan calibration_data.py + calibrate_constants.py for `yahoo_client.METHOD(`
    # and also `client.METHOD(` when YahooFantasyClient is in scope (calibrate_constants).
    pat_client_method = re.compile(r"(?:yahoo_client|client)\.([a-z_][a-zA-Z0-9_]*)\s*\(")
    called_methods: set[str] = set()
    for relpath in ("src/validation/calibration_data.py", "calibrate_constants.py"):
        p = REPO_ROOT / relpath
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        for m in pat_client_method.finditer(text):
            called_methods.add(m.group(1))

    # Exclude things that are obviously not YahooFantasyClient calls
    # (e.g. logger.error, conn.execute -- those won't match yahoo_client/client
    # patterns above, so the regex already filters them. But `client.save()` etc.
    # in other contexts may collide with ConstantSet's `client` named var, so
    # we keep the regex narrow to yahoo_client + client-after-YahooFantasyClient
    # construction.)

    missing = called_methods - client_methods
    assert not missing, (
        f"BUG-016 regression: calibration code calls method(s) {missing} "
        f"that do not exist on YahooFantasyClient. "
        f"Available client methods (first 30): {sorted(client_methods)[:30]}"
    )


def test_calibrate_constants_constructs_client_correctly():
    """calibrate_constants.py must pass a league_id to YahooFantasyClient
    (or use a factory that fills it in from env). Naive YahooFantasyClient()
    raises TypeError on missing required positional arg."""
    p = REPO_ROOT / "calibrate_constants.py"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    bad_calls: list[str] = []
    lines = text.splitlines()
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "YahooFantasyClient(" in line:
            # Gather the full call (could span multiple lines)
            full = stripped
            if not full.rstrip().endswith(")"):
                for i in range(line_no, min(line_no + 4, len(lines))):
                    full += " " + lines[i].strip()
                    if ")" in lines[i]:
                        break
            if "league_id" not in full and "()" in full:
                bad_calls.append(f"line {line_no}: {full[:120]}")
    assert not bad_calls, f"BUG-016 regression: YahooFantasyClient() called without league_id. Offenders: {bad_calls}"
