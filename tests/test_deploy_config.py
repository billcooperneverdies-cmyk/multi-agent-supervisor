"""Deployment contract tests: BUG-005/006/009/021/022/024/025.

Assert the docker/compose/env invariants that previously shipped footguns:

- BUG-005/006: required secrets use :? (hard fail, no default anywhere) and
  are absent as defaults in code/compose/env-template.
- BUG-009: Dockerfile.agent default CMD serves the FastAPI app the
  healthcheck probes; CLI available explicitly.
- BUG-021: GITHUB_TOKEN/SLACK_BOT_TOKEN are not plumbed into litellm-proxy.
- BUG-022: the agent image actually contains mcp_server/ AND agents/.
- BUG-024: the dead redis service is gone.
- BUG-025: no standalone mcp-server service in compose; stdio transport is
  in-process via MCP_SERVER_CMD.
"""

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path):
    with open(os.path.join(REPO_ROOT, path)) as fh:
        return fh.read()


COMPOSE = _read("docker-compose.yml")
DOCKERFILE_AGENT = _read("Dockerfile.agent")
DOCKERFILE_MCP = _read("Dockerfile.mcp")
ENV_EXAMPLE = _read(".env.example")
STATE_MACHINE = _read("orchestrator/state_machine.py")


def _service_block(compose: str, name: str) -> str:
    """Extract the YAML block of a compose service (indentation-based)."""
    pattern = re.compile(rf"^  {name}:\n((?:    .*\n|\s*\n)*)", re.MULTILINE)
    match = pattern.search(compose)
    assert match, f"service {name!r} not found in docker-compose.yml"
    return match.group(1)


# ── BUG-005: LITELLM_MASTER_KEY / LITELLM_API_KEY required, no defaults ──


def test_litellm_master_key_required_no_default():
    assert "${LITELLM_MASTER_KEY:?" in COMPOSE
    assert "sk-litellm-master-key" not in COMPOSE


def test_litellm_api_key_required_no_default():
    assert "${LITELLM_API_KEY:?" in COMPOSE
    assert "sk-litellm-master-key" not in STATE_MACHINE
    assert "sk-litellm-master-key" not in ENV_EXAMPLE


def test_get_llm_fails_fast_without_api_key():
    """The code must refuse to build an LLM client with no key configured."""
    import os as _os

    _os.environ.pop("LITELLM_API_KEY", None)
    from orchestrator import state_machine as sm

    sm.reset_llm_factory()
    sm.reset_llm_caches()
    try:
        try:
            sm.get_llm()
        except RuntimeError as exc:
            assert "LITELLM_API_KEY" in str(exc)
        else:
            raise AssertionError("get_llm() without LITELLM_API_KEY must fail fast")
    finally:
        sm.reset_llm_caches()


# ── BUG-006: POSTGRES_PASSWORD required; db not host-published ───────────


def test_postgres_password_required_no_default():
    assert "${POSTGRES_PASSWORD:?" in COMPOSE
    assert "POSTGRES_PASSWORD:-postgres" not in COMPOSE


def test_postgres_port_not_published_on_host():
    postgres = _service_block(COMPOSE, "postgres")
    assert re.search(r"^\s+ports:", postgres, re.MULTILINE) is None


# ── BUG-009: agent Dockerfile CMD/healthcheck coherence ──────────────────


def test_agent_dockerfile_cmd_serves_api_and_healthcheck_matches():
    assert "uvicorn orchestrator.api:app" in DOCKERFILE_AGENT
    assert "/health" in DOCKERFILE_AGENT
    assert 'CMD ["python", "-m", "orchestrator.agent"]' not in DOCKERFILE_AGENT


def test_agent_dockerfile_documents_cli_entry():
    assert "python -m orchestrator.agent" in DOCKERFILE_AGENT


# ── BUG-021: tool tokens not plumbed into the litellm proxy ──────────────


def test_tool_tokens_absent_from_litellm_proxy_service():
    proxy = _service_block(COMPOSE, "litellm-proxy")
    assert "GITHUB_TOKEN" not in proxy
    assert "SLACK_BOT_TOKEN" not in proxy


def test_tool_tokens_present_in_agent_service():
    agent = _service_block(COMPOSE, "agent")
    assert "GITHUB_TOKEN" in agent
    assert "SLACK_BOT_TOKEN" in agent


# ── BUG-022: agent image contains mcp_server and agents ──────────────────


def test_agent_dockerfile_copies_mcp_server_and_agents():
    assert "COPY mcp_server/" in DOCKERFILE_AGENT
    assert "COPY agents/" in DOCKERFILE_AGENT
    assert "COPY orchestrator/" in DOCKERFILE_AGENT


# ── BUG-024: dead redis service removed ──────────────────────────────────


def test_no_redis_service_anywhere():
    assert "redis" not in COMPOSE.lower()


# ── BUG-025: no standalone mcp-server service; stdio stays in-process ────


def test_no_standalone_mcp_server_service_in_compose():
    assert re.search(r"^  mcp-server:", COMPOSE, re.MULTILINE) is None
    assert "Dockerfile.mcp" not in COMPOSE


def test_agent_sets_mcp_server_cmd():
    agent = _service_block(COMPOSE, "agent")
    assert "MCP_SERVER_CMD" in agent


def test_mcp_dockerfile_documents_stdio_only():
    assert "stdio" in DOCKERFILE_MCP
