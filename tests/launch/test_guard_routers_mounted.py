"""Guard: every api/routers/*.py that defines a `router` is imported AND
included in api/main.py. Catches the 'feature exists in code but is not
mounted / not in the contract' drift (Codex Phase 0 inventory guard)."""

from __future__ import annotations

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ROUTERS_DIR = _ROOT / "api" / "routers"
_MAIN = _ROOT / "api" / "main.py"


def _router_modules_on_disk() -> set[str]:
    names: set[str] = set()
    for p in _ROUTERS_DIR.glob("*.py"):
        if p.name == "__init__.py":
            continue
        text = p.read_text(encoding="utf-8")
        # Only count modules that actually define a top-level `router`.
        if re.search(r"^router\s*=", text, re.MULTILINE):
            names.add(p.stem)
    return names


def _router_modules_imported_in_main() -> set[str]:
    text = _MAIN.read_text(encoding="utf-8")
    return set(re.findall(r"from api\.routers\.(\w+) import router", text))


def _include_router_call_count() -> int:
    text = _MAIN.read_text(encoding="utf-8")
    return len(re.findall(r"app\.include_router\(", text))


def test_every_router_module_is_imported_in_main():
    on_disk = _router_modules_on_disk()
    imported = _router_modules_imported_in_main()
    missing = on_disk - imported
    assert not missing, (
        "api/routers modules define a `router` but are not imported in api/main.py: "
        f"{sorted(missing)}. Mount them or remove the dead router."
    )


def test_import_and_include_counts_match():
    # Every imported router alias should also be passed to include_router.
    imported = _router_modules_imported_in_main()
    includes = _include_router_call_count()
    assert includes >= len(imported), (
        f"{len(imported)} routers imported but only {includes} include_router(...) calls "
        "in api/main.py — a router was imported but never mounted."
    )
