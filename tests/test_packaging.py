"""Packaging contract tests for BUG-003, BUG-004, BUG-028, BUG-044.

These tests codify the packaging/dependency rules that must hold:

- BUG-003: hatchling must be told which packages ship in the wheel
  (orchestrator, mcp_server, agents) so ``pip install -e .`` works.
- BUG-004: ``mcp`` must be pinned ``>=1.28,<2`` — mcp 2.x removed
  ``mcp.server.fastmcp`` which ``mcp_server/server.py`` imports.
- BUG-028: ``fastapi`` and ``uvicorn`` are imported by
  ``orchestrator/api.py`` and must be declared as runtime dependencies.
- BUG-044: supply-chain hygiene — every runtime dependency carries both a
  floor and an upper-bound ceiling, and no Docker/Compose image reference
  uses a moving tag (``:latest``, ``main-latest``, bare ``python:3.12-slim``,
  floating ``redis:7-alpine``).
"""

import os
import re
import tomllib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPROJECT = os.path.join(REPO_ROOT, "pyproject.toml")
DOCKERFILES = [
    os.path.join(REPO_ROOT, "Dockerfile.agent"),
    os.path.join(REPO_ROOT, "Dockerfile.mcp"),
]
COMPOSE = os.path.join(REPO_ROOT, "docker-compose.yml")

_SPEC_RE = re.compile(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(.*)$")
_BOUND_RE = re.compile(r"(>=|<=|==|~=|>|<)\s*([0-9][A-Za-z0-9.*]*)")


def _load_pyproject():
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def _runtime_deps():
    return _load_pyproject()["project"]["dependencies"]


def _parse_dep(dep):
    """Return (name, [(op, version_tuple), ...])."""
    m = _SPEC_RE.match(dep.strip())
    assert m, f"unparseable dependency spec: {dep!r}"
    name = m.group(1).lower()
    bounds = []
    for op, ver in _BOUND_RE.findall(m.group(3) or ""):
        parts = []
        for piece in ver.split("."):
            if piece == "*":
                break
            digits = re.match(r"\d+", piece)
            parts.append(int(digits.group()) if digits else 0)
        bounds.append((op, tuple(parts)))
    return name, bounds


def _floor(name, bounds):
    floors = [v for op, v in bounds if op in (">=", "==", "~=")]
    assert floors, f"{name}: missing lower bound (>=) in spec"
    return max(floors)


def _ceiling(name, bounds):
    ceilings = [v for op, v in bounds if op in ("<", "<=")]
    assert ceilings, f"{name}: missing upper bound (<) ceiling in spec"
    return min(ceilings)


# ── BUG-003: hatchling wheel package mapping ─────────────────────────────


def test_wheel_packages_declared():
    pyproject = _load_pyproject()
    wheel_cfg = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
    )
    packages = wheel_cfg.get("packages", [])
    assert packages, "BUG-003: [tool.hatch.build.targets.wheel] packages missing"
    for pkg in ("orchestrator", "mcp_server", "agents"):
        assert pkg in packages, f"BUG-003: wheel packages missing {pkg!r}"


# ── BUG-004: mcp pinned below 2.x ─────────────────────────────────────────


def test_mcp_pinned_below_2():
    deps = dict(_parse_dep(d) for d in _runtime_deps())
    assert "mcp" in deps, "mcp dependency missing"
    bounds = deps["mcp"]
    floor = _floor("mcp", bounds)
    ceiling = _ceiling("mcp", bounds)
    assert floor >= (1, 28), f"BUG-004: mcp floor {floor} below verified 1.28"
    assert ceiling < (3,), f"BUG-004: mcp ceiling {ceiling} allows breaking major"
    assert ceiling <= (2,), f"BUG-004: mcp ceiling {ceiling} must exclude 2.x (fastmcp removed)"


# ── BUG-028: fastapi + uvicorn declared ───────────────────────────────────


def test_fastapi_declared():
    names = {_parse_dep(d)[0] for d in _runtime_deps()}
    assert "fastapi" in names, "BUG-028: fastapi imported but undeclared"


def test_uvicorn_declared():
    names = {_parse_dep(d)[0] for d in _runtime_deps()}
    assert "uvicorn" in names, "BUG-028: uvicorn imported but undeclared"


# ── BUG-044: floor+ceiling on every runtime dep, pinned image tags ────────


def test_every_runtime_dep_has_floor_and_ceiling():
    for dep in _runtime_deps():
        name, bounds = _parse_dep(dep)
        _floor(name, bounds)
        _ceiling(name, bounds)


def test_every_dev_dep_has_floor_and_ceiling():
    pyproject = _load_pyproject()
    for dep in pyproject["project"]["optional-dependencies"]["dev"]:
        name, bounds = _parse_dep(dep)
        _floor(name, bounds)
        _ceiling(name, bounds)


def _dockerfile_contents():
    return {os.path.basename(p): open(p).read() for p in DOCKERFILES}


def test_dockerfiles_no_moving_base_tags():
    for fname, text in _dockerfile_contents().items():
        for line in text.splitlines():
            if not line.strip().upper().startswith("FROM"):
                continue
            assert ":latest" not in line, f"BUG-044: {fname} uses :latest ({line.strip()})"
            assert "python:3.12-slim " not in line and line.strip() != "FROM python:3.12-slim", (
                f"BUG-044: {fname} uses unpinned python:3.12-slim bare tag"
            )
            assert "python:3.12-slim\n" not in line
            assert "FROM python:3.12-slim" not in line or "bookworm" in line or "@sha256:" in line, (
                f"BUG-044: {fname} base image must be variant- or digest-pinned"
            )


def test_compose_no_moving_image_tags():
    text = open(COMPOSE).read()
    assert ":latest" not in text, "BUG-044: docker-compose.yml still references :latest"
    assert "main-latest" not in text, "BUG-044: litellm image still on moving main-latest tag"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("image:"):
            ref = stripped.split("image:", 1)[1].strip()
            assert "@sha256:" in ref or re.search(r":[A-Za-z0-9][A-Za-z0-9._\-]*[0-9][A-Za-z0-9._\-]*$", ref), (
                f"BUG-044: compose image not pinned to a stable tag/digest: {ref}"
            )
