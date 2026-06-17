"""Guard: api/routers/* must stay thin — no analytics math, no engine imports.
All data work belongs in api/services/. Mirrors the pages/ logic-free rule."""

import ast
import pathlib

ROUTERS = pathlib.Path(__file__).resolve().parents[2] / "api" / "routers"


def test_routers_do_not_import_src_engines():
    offenders = []
    for py in ROUTERS.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
                offenders.append(f"{py.name}: from {node.module}")
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name.startswith("src."):
                        offenders.append(f"{py.name}: import {n.name}")
    assert not offenders, "routers must not import src.* engines directly — use api/services/: " + "; ".join(offenders)


def test_routers_have_no_arithmetic_assignments():
    # crude but effective: no BinOp on the RHS of an assignment in a router
    offenders = []
    for py in ROUTERS.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
                offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, "no arithmetic in routers (push math into services/engines): " + "; ".join(offenders)
