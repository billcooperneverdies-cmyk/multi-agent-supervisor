"""Graph-core regression tests (BUG-002/011/012/013/015/018/030/031/032/033/034/038,
plus the graph-side of BUG-008/010/014).

All tests run fully offline through the ScriptedChatModel seam (see conftest).
"""

import asyncio
import importlib
import sys

import pytest
from conftest import initial_state
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


def run(coro):
    return asyncio.run(coro)


def build_graph():
    from orchestrator.state_machine import build_team_workflow

    return build_team_workflow(checkpointer=InMemorySaver())


# ───────────────────────────────────────────────────────
# BUG-002 / BUG-031: graph compiles at import; correct annotation
# ───────────────────────────────────────────────────────

def test_graph_compiles_at_import():
    """BUG-002: `Annotated[list[str], list]` was an invalid reducer; importing
    orchestrator.state_machine must succeed and expose a compiled graph."""
    sys.modules.pop("orchestrator.state_machine", None)
    module = importlib.import_module("orchestrator.state_machine")
    from langgraph.graph.state import CompiledStateGraph

    assert isinstance(module.team_graph, CompiledStateGraph)


def test_build_team_workflow_returns_compiled_graph():
    """BUG-031: build_team_workflow returns a CompiledStateGraph."""
    from langgraph.graph.state import CompiledStateGraph

    from orchestrator.state_machine import build_team_workflow

    graph = build_team_workflow(checkpointer=InMemorySaver())
    assert isinstance(graph, CompiledStateGraph)
    assert hasattr(graph, "ainvoke")


# ───────────────────────────────────────────────────────
# BUG-011 / BUG-012: sentinel routing, no substring routing,
# termination evaluated before agent-name tokens
# ───────────────────────────────────────────────────────

class TestSentinelRouting:
    def test_free_text_never_terminates(self):
        """'end' inside 'frontend'/'backend'/'recommend' must not terminate."""
        from orchestrator.state_machine import alice_route

        for text in [
            "Setup the backend skeleton",
            "Let's build the frontend first",
            "I recommend we end-to-end test later",
            "All done, thanks Bob",  # free text naming an agent is NOT routing
        ]:
            assert alice_route(text, {}) != "end", text

    def test_free_text_never_routes_to_agents(self):
        """Substring hits like 'qa' in 'Qatar' must not route."""
        from orchestrator.state_machine import alice_route

        assert alice_route("The Qatar office asked about QA processes", {}) != "diana"

    def test_free_text_never_passes_qa(self):
        """BUG-011: 'pass' in 'password' must not produce a QA pass verdict."""
        from orchestrator.state_machine import diana_route

        assert diana_route("No issues found, password hashing looks fine") != "qa_pass"

    def test_whole_token_sentinels_route(self):
        from orchestrator.state_machine import alice_route, diana_route

        assert alice_route("Assigning UI work. [[NEXT:bob]]", {}) == "bob"
        assert alice_route("Backend next. [[NEXT:charlie]]", {}) == "charlie"
        assert alice_route("Time for review. [[NEXT:diana]]", {}) == "diana"
        assert alice_route("Need clarification. [[NEXT:human]]", {}) == "human"
        assert alice_route("Project complete. [[END]]", {}) == "end"
        assert diana_route("Looks great. [[QA:pass]]") == "qa_pass"
        assert diana_route("Broken. [[QA:fail]] [[NEXT:bob]]") == "bob"
        assert diana_route("Broken. [[QA:fail]] [[NEXT:charlie]]") == "charlie"

    def test_termination_beats_agent_names(self):
        """BUG-012: termination tokens are evaluated BEFORE agent-name tokens."""
        from orchestrator.state_machine import alice_route, diana_route

        assert alice_route("[[END]] [[NEXT:bob]]", {}) == "end"
        assert alice_route("Done [[NEXT:diana]] ... [[END]]", {}) == "end"
        # A QA verdict decides the outcome; agent-name tokens only pick the fixer.
        assert diana_route("[[QA:pass]] [[NEXT:bob]]") == "qa_pass"
        assert diana_route("[[QA:fail]] [[NEXT:bob]]") == "bob"

    def test_case_insensitive_and_whitespace_tolerant(self):
        from orchestrator.state_machine import alice_route

        assert alice_route("ok [[ next : BOB ]]", {}) == "bob"
        assert alice_route("done [[end]]", {}) == "end"


# ───────────────────────────────────────────────────────
# BUG-013: persona prompt factories reach the LLM
# ───────────────────────────────────────────────────────

def test_persona_system_prompts_reach_llm(install_model):
    """BUG-013: each node's outbound messages must carry its agents/*.py persona."""
    model = install_model([
        "Plan made. [[NEXT:bob]]",
        "UI built. [[NEXT:alice]]",
        "Everything done. [[END]]",
    ])
    graph = build_graph()
    final = run(graph.ainvoke(initial_state(), config={"configurable": {"thread_id": "p1"}}))
    assert final["active_agent"] == "end"
    assert len(model.received) == 3
    alice_prompt, bob_prompt, _ = model.received
    assert any("项目经理" in str(m.content) for m in alice_prompt), "Alice persona missing"
    assert any("前端" in str(m.content) for m in bob_prompt), "Bob persona missing"
    # The routing protocol itself must be part of the persona prompt.
    assert any("[[NEXT:" in str(m.content) for m in alice_prompt)


# ───────────────────────────────────────────────────────
# BUG-015: async nodes call ainvoke (no executor starvation)
# ───────────────────────────────────────────────────────

def test_nodes_use_async_ainvoke(install_model):
    model = install_model(["[[END]]"])
    graph = build_graph()
    run(graph.ainvoke(initial_state(), config={"configurable": {"thread_id": "a1"}}))
    assert model.async_calls >= 1
    assert model.sync_calls == 0, "sync .invoke path used — nodes must be async"


# ───────────────────────────────────────────────────────
# BUG-018: list-type content normalization
# ───────────────────────────────────────────────────────

def test_normalize_content_handles_anthropic_blocks():
    from orchestrator.state_machine import normalize_content

    blocks = [
        {"type": "text", "text": "Review complete. "},
        {"type": "tool_use", "id": "x", "name": "t", "input": {}},
        {"type": "text", "text": "[[QA:pass]]"},
    ]
    text = normalize_content(blocks)
    assert "Review complete." in text
    assert "[[QA:pass]]" in text
    assert normalize_content("plain") == "plain"


def test_node_survives_list_content(install_model):
    """BUG-018: AIMessage.content as list-of-blocks must not crash routing."""
    install_model([
        AIMessage(content=[{"type": "text", "text": "All wrapped up. [[END]]"}]),
    ])
    graph = build_graph()
    final = run(graph.ainvoke(initial_state(), config={"configurable": {"thread_id": "l1"}}))
    assert final["active_agent"] == "end"


# ───────────────────────────────────────────────────────
# BUG-030: explicit recursion_limit coherent with max_iterations
# ───────────────────────────────────────────────────────

def test_make_run_config_recursion_limit():
    from orchestrator.state_machine import make_run_config

    cfg = make_run_config("task-1", 20)
    assert cfg["recursion_limit"] == 2 * 20 + 10
    assert cfg["configurable"]["thread_id"] == "task-1"
    assert make_run_config("t", 50)["recursion_limit"] == 110


def test_graph_run_respects_recursion_limit_config(install_model):
    """A run configured via make_run_config completes under its recursion_limit."""
    from orchestrator.state_machine import make_run_config

    install_model(["[[END]]"])
    graph = build_graph()
    final = run(graph.ainvoke(initial_state(), config=make_run_config("rc1", 20)))
    assert final["active_agent"] == "end"


# ───────────────────────────────────────────────────────
# BUG-032: worker nodes write branch fields
# ───────────────────────────────────────────────────────

def test_bob_writes_frontend_branch(install_model):
    install_model([
        "[[NEXT:bob]]",
        "UI implemented. [[NEXT:alice]]",
        "[[END]]",
    ])
    graph = build_graph()
    final = run(graph.ainvoke(initial_state(), config={"configurable": {"thread_id": "b1"}}))
    assert final["frontend_branch"], "bob_node must write frontend_branch"
    assert final["backend_branch"] is None


def test_charlie_writes_backend_branch(install_model):
    install_model([
        "[[NEXT:charlie]]",
        "API implemented. [[NEXT:alice]]",
        "[[END]]",
    ])
    graph = build_graph()
    final = run(graph.ainvoke(initial_state(), config={"configurable": {"thread_id": "c1"}}))
    assert final["backend_branch"], "charlie_node must write backend_branch"


def test_alice_fallback_uses_branch_state():
    """BUG-032: with frontend done and backend missing, fallback routes to charlie."""
    from orchestrator.state_machine import alice_route

    assert (
        alice_route("no sentinel here", {"frontend_branch": "f", "backend_branch": None})
        == "charlie"
    )
    assert (
        alice_route("no sentinel here", {"frontend_branch": None, "backend_branch": "b"})
        == "bob"
    )


# ───────────────────────────────────────────────────────
# BUG-033 / BUG-034: SecretStr + cached LLM client
# ───────────────────────────────────────────────────────

def test_get_llm_caches_singleton_per_config(monkeypatch):
    from orchestrator.state_machine import get_llm, reset_llm_caches

    # BUG-005: LITELLM_API_KEY is now required (no hardcoded default).
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test-dummy-key")
    reset_llm_caches()
    llm1 = get_llm()
    llm2 = get_llm()
    assert llm1 is llm2, "get_llm must cache one client per config"
    reset_llm_caches()


def test_get_llm_api_key_is_secretstr(monkeypatch):
    from pydantic import SecretStr

    from orchestrator.state_machine import get_llm, reset_llm_caches

    # BUG-005: LITELLM_API_KEY is now required (no hardcoded default).
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test-dummy-key")
    reset_llm_caches()
    assert isinstance(get_llm().openai_api_key, SecretStr)
    reset_llm_caches()


# ───────────────────────────────────────────────────────
# BUG-038: bounded history window per LLM call
# ───────────────────────────────────────────────────────

def test_history_window_bounds_prompt(install_model, monkeypatch):
    monkeypatch.setenv("HISTORY_WINDOW", "20")
    model = install_model(["[[END]]"])
    history = [HumanMessage(content="TASK-ROOT")] + [
        HumanMessage(content=f"noise-{i}") for i in range(40)
    ]
    graph = build_graph()
    run(
        graph.ainvoke(
            initial_state(messages=history), config={"configurable": {"thread_id": "h1"}}
        )
    )
    outbound = model.received[0]
    contents = [str(m.content) for m in outbound]
    assert any("TASK-ROOT" in c for c in contents), "first task message must be kept"
    assert not any("noise-0" in c for c in contents), "oldest noise must be dropped"
    # system prompt + first task message + last 20
    assert len(outbound) <= 1 + 1 + 20


# ───────────────────────────────────────────────────────
# BUG-008 / BUG-010 (graph side): interrupt + Command(resume), no stdin
# ───────────────────────────────────────────────────────

def test_human_node_interrupts_and_resumes_with_stdin_closed(install_model, monkeypatch):
    """BUG-008/010: graph pauses at the human node via interrupt() and resumes
    via Command(resume=...). No stdin may be touched anywhere."""
    def _no_stdin(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("stdin input() used inside graph run")

    monkeypatch.setattr("builtins.input", _no_stdin)
    install_model([
        "Requirements unclear. [[NEXT:human]]",
        "Thanks, all clear now. [[END]]",
    ])
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "hitl-1"}}

    paused = run(graph.ainvoke(initial_state(), config=cfg))
    assert "__interrupt__" in paused, "graph must pause with an interrupt payload"
    assert paused["pending_human"] is True

    resumed = run(graph.ainvoke(Command(resume="approved, proceed"), config=cfg))
    assert "__interrupt__" not in resumed
    assert resumed["active_agent"] == "end"
    assert resumed["human_input"] == "approved, proceed"
    assert resumed["pending_human"] is False


def test_checkpoint_state_survives_between_invocations(install_model):
    """BUG-010: checkpointer persists state; get_state exposes the pause."""
    install_model(["[[NEXT:human]]", "[[END]]"])
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "hitl-2"}}
    run(graph.ainvoke(initial_state(), config=cfg))
    snap = run(graph.aget_state(cfg))
    assert snap.values.get("pending_human") is True
    assert snap.next, "interrupted graph must have a pending next node"


# ───────────────────────────────────────────────────────
# BUG-014: MCP tool wiring (graceful degradation + tool results in state)
# ───────────────────────────────────────────────────────

def test_mcp_unset_graph_runs_without_tools(install_model):
    """BUG-014: MCP_SERVER_CMD unset -> no tools bound, graph still runs."""
    model = install_model(["[[END]]"])
    graph = build_graph()
    final = run(graph.ainvoke(initial_state(), config={"configurable": {"thread_id": "m1"}}))
    assert final["active_agent"] == "end"
    assert model.bound_tools == []
    assert not final.get("tool_results")


def test_mcp_tools_loaded_when_cmd_set(monkeypatch):
    """BUG-014: MCP_SERVER_CMD spawns a stdio MCP server and tools are bound."""
    pytest.importorskip("mcp")
    from orchestrator import state_machine as sm

    async def _load():
        # The repo's MCP server validates these at startup (values unused here).
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("MCP_SERVER_CMD", f"{sys.executable} -m mcp_server.main")
        sm.reset_mcp_tools()
        try:
            return await sm.get_mcp_tools()
        finally:
            sm.reset_mcp_tools()

    tools = run(_load())
    assert tools, "expected real tools from the repo's MCP server over stdio"


def test_tool_results_and_urls_land_in_state(install_model, monkeypatch):
    """BUG-014: tool calls execute, ToolMessages + tool_results + pr_urls recorded."""
    from langchain_core.tools import tool

    from orchestrator import state_machine as sm

    @tool
    def github_create_pr(title: str) -> str:
        """Create a GitHub pull request."""
        return f"created https://github.com/octo/demo/pull/42 for {title}"

    async def fake_tools():
        return [github_create_pr]

    monkeypatch.setattr(sm, "get_mcp_tools", fake_tools)
    install_model([
        "[[NEXT:bob]]",
        AIMessage(
            content="",
            tool_calls=[{"name": "github_create_pr", "args": {"title": "UI"}, "id": "call-1"}],
        ),
        "PR opened. [[NEXT:alice]]",
        "[[END]]",
    ])
    graph = build_graph()
    final = run(graph.ainvoke(initial_state(), config={"configurable": {"thread_id": "m2"}}))
    assert final["active_agent"] == "end"
    assert "https://github.com/octo/demo/pull/42" in final["pr_urls"]
    assert final["tool_results"], "tool_results must be populated"
    assert any(isinstance(m, ToolMessage) for m in final["messages"])
