"""Deploy-config & secrets-hygiene tests.

Covers BUG-005 (hardcoded LiteLLM key), BUG-006 (default postgres creds /
published 5432), BUG-009 (agent image runs the API, not the CLI), BUG-021
(token sprawl into litellm-proxy), BUG-022 (.env.example completeness),
BUG-024 (dead services/vars) and BUG-025 (MCP stdio sidecar -> in-process
spawn by the agent service).

Everything here is static (repo-file) analysis plus a small code-level
fail-fast check — no docker daemon required. The compose file is parsed with
PyYAML (docker is unavailable in CI/dev containers).
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
DOCKERFILE_AGENT_PATH = REPO_ROOT / "Dockerfile.agent"
DOCKERFILE_MCP_PATH = REPO_ROOT / "Dockerfile.mcp"
LITELLM_CONFIG_PATH = REPO_ROOT / "configs" / "litellm_proxy.yaml"

CODE_DIRS = ("orchestrator", "mcp_server", "agents")

# Built by concatenation so this test file itself never contains the literal.
HARD_CODED_KEY = "sk-litellm-" + "master-key"

COMPOSE_TEXT = COMPOSE_PATH.read_text(encoding="utf-8")
COMPOSE = yaml.safe_load(COMPOSE_TEXT)
ENV_EXAMPLE_TEXT = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
DOCKERFILE_AGENT = DOCKERFILE_AGENT_PATH.read_text(encoding="utf-8")
DOCKERFILE_MCP = DOCKERFILE_MCP_PATH.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _service_env(compose: dict, service: str) -> dict:
    """Return a service's environment as a dict (handles list or mapping form)."""
    env = compose["services"][service].get("environment") or {}
    if isinstance(env, dict):
        return dict(env)
    out = {}
    for item in env:
        key, _, value = item.partition("=")
        out[key] = value
    return out


def _uncommented_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def _code_env_vars() -> set[str]:
    """Every env var read via os.environ/os.getenv in shipping code."""
    keys: set[str] = set()
    literal = re.compile(r"""os\.(?:environ\.get|getenv)\(\s*['"]([A-Z][A-Z0-9_]*)['"]""")
    subscript = re.compile(r"""os\.environ\[\s*['"]([A-Z][A-Z0-9_]*)['"]\s*\]""")
    indirect_call = re.compile(r"""os\.(?:environ\.get|getenv)\(\s*([A-Z][A-Z0-9_]*)\s*[,)]""")
    constant = re.compile(r"""^([A-Z][A-Z0-9_]*)\s*=\s*['"]([A-Z][A-Z0-9_]*)['"]""", re.M)
    for dirname in CODE_DIRS:
        for path in (REPO_ROOT / dirname).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            keys.update(literal.findall(text))
            keys.update(subscript.findall(text))
            constants = dict(constant.findall(text))
            for name in indirect_call.findall(text):
                if name in constants:  # e.g. os.environ.get(HISTORY_WINDOW_ENV, ...)
                    keys.add(constants[name])
    return keys


def _compose_interpolated_vars() -> set[str]:
    pattern = re.compile(r"\$\{([A-Z][A-Z0-9_]*)[^}]*\}")
    keys: set[str] = set()
    for line in _uncommented_lines(COMPOSE_TEXT):
        keys.update(pattern.findall(line))
    return keys


def _litellm_config_env_vars() -> set[str]:
    text = LITELLM_CONFIG_PATH.read_text(encoding="utf-8")
    return set(re.findall(r"os\.environ/([A-Z][A-Z0-9_]*)", text))


def _dockerfile_env_vars() -> set[str]:
    keys: set[str] = set()
    for text in (DOCKERFILE_AGENT, DOCKERFILE_MCP):
        for line in _uncommented_lines(text):
            keys.update(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)\}?", line))
            keys.update(re.findall(r"^\s*(?:ENV|ARG)\s+([A-Z][A-Z0-9_]*)", line))
    return keys


def _env_example_documented_vars() -> set[str]:
    keys: set[str] = set()
    for line in ENV_EXAMPLE_TEXT.splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)=", line)
        if match:
            keys.add(match.group(1))
    return keys


def _all_consumed_vars() -> set[str]:
    return (
        _code_env_vars()
        | _compose_interpolated_vars()
        | _litellm_config_env_vars()
        | _dockerfile_env_vars()
    )


# ──────────────────────────────────────────────────────────────
# BUG-005 — no hardcoded LiteLLM key; fail fast without LITELLM_API_KEY
# ──────────────────────────────────────────────────────────────

def test_no_hardcoded_litellm_key_anywhere():
    offenders = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError):
            continue
        if HARD_CODED_KEY in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"hardcoded LiteLLM key found in: {offenders}"


def test_compose_requires_litellm_keys():
    assert "${LITELLM_MASTER_KEY:?" in COMPOSE_TEXT, (
        "litellm-proxy must require LITELLM_MASTER_KEY (no default)"
    )
    assert "${LITELLM_API_KEY:?" in COMPOSE_TEXT, (
        "agent service must require LITELLM_API_KEY (no default)"
    )
    assert ":-sk-" not in COMPOSE_TEXT, "compose still defaults some key to a sk- literal"


def test_get_llm_fails_fast_without_litellm_api_key(monkeypatch):
    from orchestrator import state_machine as sm

    sm.reset_llm_factory()
    sm.reset_llm_caches()
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LITELLM_API_KEY"):
        sm.get_llm()


def test_get_llm_builds_with_litellm_api_key(monkeypatch):
    from langchain_openai import ChatOpenAI

    from orchestrator import state_machine as sm

    sm.reset_llm_factory()
    sm.reset_llm_caches()
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test-dummy-key")
    llm = sm.get_llm()
    assert isinstance(llm, ChatOpenAI)
    sm.reset_llm_caches()


# ──────────────────────────────────────────────────────────────
# BUG-006 — postgres: required password, no host port
# ──────────────────────────────────────────────────────────────

def test_postgres_password_has_no_default():
    assert "${POSTGRES_PASSWORD:?" in COMPOSE_TEXT, (
        "POSTGRES_PASSWORD must use required-var interpolation"
    )
    assert "POSTGRES_PASSWORD:-" not in COMPOSE_TEXT, (
        "POSTGRES_PASSWORD still has a default value"
    )


def test_postgres_port_not_published():
    postgres = COMPOSE["services"]["postgres"]
    assert "ports" not in postgres, "postgres must not publish 5432 on the host"


# ──────────────────────────────────────────────────────────────
# BUG-009 — Dockerfile.agent serves the API; copies all packages
# ──────────────────────────────────────────────────────────────

def _dockerfile_cmd(text: str) -> str:
    """The image's default CMD (last CMD instruction; earlier ones may be part
    of a HEALTHCHECK clause)."""
    cmd_lines = [
        line for line in text.splitlines() if line.strip().upper().startswith("CMD ")
    ]
    return cmd_lines[-1] if cmd_lines else ""


def test_dockerfile_agent_copies_all_packages():
    for pkg in ("orchestrator", "agents", "mcp_server"):
        assert re.search(rf"^COPY {pkg}/ \./{pkg}/", DOCKERFILE_AGENT, re.M), (
            f"Dockerfile.agent must COPY {pkg}/"
        )


def test_dockerfile_agent_cmd_serves_api_not_cli():
    cmd = _dockerfile_cmd(DOCKERFILE_AGENT)
    assert cmd, "Dockerfile.agent has no CMD"
    assert "orchestrator.api" in cmd, "default CMD must serve the FastAPI app"
    assert "uvicorn" in cmd.lower(), "default CMD must run uvicorn"
    assert "orchestrator.agent" not in cmd, "interactive CLI must not be the default CMD"
    assert "AGENT_PORT" in cmd, "CMD must honor AGENT_PORT"


def test_dockerfile_agent_healthcheck_matches_api():
    match = re.search(r"HEALTHCHECK.*?CMD\s+(.*)", DOCKERFILE_AGENT, re.S)
    assert match, "Dockerfile.agent has no HEALTHCHECK"
    check = match.group(1)
    assert "/health" in check
    assert "AGENT_PORT" in check, "healthcheck must target the AGENT_PORT"


# ──────────────────────────────────────────────────────────────
# BUG-021 / BUG-025 — least-privilege env; MCP spawned by the agent
# ──────────────────────────────────────────────────────────────

def test_litellm_proxy_has_no_github_or_slack_tokens():
    env = _service_env(COMPOSE, "litellm-proxy")
    assert "GITHUB_TOKEN" not in env, "GITHUB_TOKEN must not reach the litellm container"
    assert "SLACK_BOT_TOKEN" not in env, "SLACK_BOT_TOKEN must not reach the litellm container"


def test_agent_service_spawns_mcp_and_holds_tokens():
    services = COMPOSE["services"]
    assert "mcp-server" not in services, (
        "standalone mcp-server sidecar must be removed (agent spawns stdio MCP in-process)"
    )
    env = _service_env(COMPOSE, "agent")
    assert env.get("MCP_SERVER_CMD") == "python -m mcp_server.main"
    assert "GITHUB_TOKEN" in env, "agent container needs GITHUB_TOKEN for spawned MCP"
    assert "SLACK_BOT_TOKEN" in env, "agent container needs SLACK_BOT_TOKEN for spawned MCP"


def test_dockerfile_mcp_documents_stdio_transport():
    assert "stdio" in DOCKERFILE_MCP.lower(), (
        "Dockerfile.mcp must document its standalone stdio/dev nature"
    )


# ──────────────────────────────────────────────────────────────
# BUG-024 — dead services removed
# ──────────────────────────────────────────────────────────────

def test_no_redis_service():
    assert "redis" not in COMPOSE["services"], "redis has zero consumers; service must go"
    volumes = COMPOSE.get("volumes") or {}
    assert "redis_data" not in volumes


# ──────────────────────────────────────────────────────────────
# BUG-022 — .env.example documents every consumed var, no phantoms
# ──────────────────────────────────────────────────────────────

def test_env_example_documents_every_consumed_var():
    consumed = _all_consumed_vars()
    documented = _env_example_documented_vars()
    missing = consumed - documented
    assert not missing, f"consumed but undocumented in .env.example: {sorted(missing)}"


def test_env_example_has_no_phantom_vars():
    consumed = _all_consumed_vars()
    documented = _env_example_documented_vars()
    phantom = documented - consumed
    assert not phantom, f"documented in .env.example but nothing consumes: {sorted(phantom)}"


def test_env_example_sanity_required_secrets_present():
    documented = _env_example_documented_vars()
    for var in ("LITELLM_MASTER_KEY", "LITELLM_API_KEY", "POSTGRES_PASSWORD"):
        assert var in documented
