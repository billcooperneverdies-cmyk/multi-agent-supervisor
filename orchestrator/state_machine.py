"""Multi-Agent LangGraph State Machine — Alice, Bob, Charlie, Diana Collaboration.

四代理协作软件开发团队工作流：
- Alice (PM): 需求分析 → 任务分配 → 进度跟踪 → 验收
- Bob (Frontend): 接收任务 → 实现 UI → 提交代码 → 通知 QA
- Charlie (Backend): 接收任务 → 实现 API → 提交代码 → 通知 QA
- Diana (QA): 接收审查任务 → 代码审查 → Bug 报告 → 验收批准
"""

from typing import Annotated, Optional, Literal, List
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from agents import (
    create_alice_prompt,
    create_bob_prompt,
    create_charlie_prompt,
    create_diana_prompt,
)


# ───────────────────────────────────────────────────────
# AgentState — 多代理共享状态
# ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    """共享状态，由所有代理节点和工具节点读写。"""

    messages: Annotated[list[AnyMessage], add_messages]
    active_agent: Literal["alice", "bob", "charlie", "diana", "human", "end"]
    task_id: Optional[str]
    task_description: Optional[str]
    project_board: Optional[dict]
    slack_thread_ts: Optional[str]
    slack_channel: Optional[str]
    repo_owner: Optional[str]
    repo_name: Optional[str]
    current_branch: Optional[str]
    frontend_branch: Optional[str]
    backend_branch: Optional[str]
    pr_urls: Annotated[list[str], list]
    issue_urls: Annotated[list[str], list]
    tool_calls: Optional[list[dict]]
    tool_results: Optional[list[dict]]
    iteration_count: int
    max_iterations: int
    human_input: Optional[str]
    pending_human: bool


# ───────────────────────────────────────────────────────
# LLM 初始化
# ───────────────────────────────────────────────────────
def get_llm(model: str = "gpt-4o") -> ChatOpenAI:
    """通过 LiteLLM 代理获取 LLM。"""
    import os
    return ChatOpenAI(
        model=model,
        api_key=os.environ.get("LITELLM_API_KEY", "sk-litellm-master-key"),
        base_url=f"{os.environ.get('LITELLM_API_BASE', 'http://localhost:4000')}/v1",
        temperature=0.2,
    )


# ───────────────────────────────────────────────────────
# 节点函数
# ───────────────────────────────────────────────────────

def alice_node(state: AgentState) -> dict:
    """Alice (PM) 节点：需求分析、任务分解、路由决策。"""
    messages = state["messages"]
    iteration = state.get("iteration_count", 0)
    
    llm = get_llm().bind_tools([])
    prompt = create_alice_prompt()
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="你是项目经理 Alice。")] + messages
    
    response = llm.invoke(messages)
    content = response.content.lower() if hasattr(response, "content") else ""
    
    if "bob" in content or "前端" in content:
        next_agent = "bob"
    elif "charlie" in content or "后端" in content:
        next_agent = "charlie"
    elif "diana" in content or "qa" in content or "测试" in content:
        next_agent = "diana"
    elif "human" in content or "ask_human" in content or "人类" in content:
        next_agent = "human"
    elif "完成" in content or "done" in content or "end" in content:
        next_agent = "end"
    else:
        if state.get("frontend_branch") and not state.get("backend_branch"):
            next_agent = "charlie"
        elif state.get("backend_branch") and not state.get("frontend_branch"):
            next_agent = "bob"
        else:
            next_agent = "bob"
    
    return {
        "messages": [response],
        "active_agent": next_agent,
        "iteration_count": iteration + 1,
    }


def bob_node(state: AgentState) -> dict:
    """Bob (Frontend) 节点：实现前端代码。"""
    messages = state["messages"]
    iteration = state.get("iteration_count", 0)
    
    llm = get_llm().bind_tools([])
    prompt = create_bob_prompt()
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="你是前端开发 Bob。")] + messages
    
    response = llm.invoke(messages)
    content = response.content.lower() if hasattr(response, "content") else ""
    
    if "完成" in content or "done" in content or "ready" in content:
        return {"messages": [response], "active_agent": "alice", "iteration_count": iteration + 1}
    
    if "ask_human" in content or "确认" in content:
        return {"messages": [response], "active_agent": "human", "pending_human": True, "iteration_count": iteration + 1}
    
    return {"messages": [response], "active_agent": "bob", "iteration_count": iteration + 1}


def charlie_node(state: AgentState) -> dict:
    """Charlie (Backend) 节点：实现后端代码。"""
    messages = state["messages"]
    iteration = state.get("iteration_count", 0)
    
    llm = get_llm().bind_tools([])
    prompt = create_charlie_prompt()
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="你是后端开发 Charlie。")] + messages
    
    response = llm.invoke(messages)
    content = response.content.lower() if hasattr(response, "content") else ""
    
    if "完成" in content or "done" in content or "ready" in content:
        return {"messages": [response], "active_agent": "alice", "iteration_count": iteration + 1}
    
    if "ask_human" in content or "确认" in content:
        return {"messages": [response], "active_agent": "human", "pending_human": True, "iteration_count": iteration + 1}
    
    return {"messages": [response], "active_agent": "charlie", "iteration_count": iteration + 1}


def diana_node(state: AgentState) -> dict:
    """Diana (QA) 节点：代码审查和验收。"""
    messages = state["messages"]
    iteration = state.get("iteration_count", 0)
    
    llm = get_llm().bind_tools([])
    prompt = create_diana_prompt()
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="你是 QA 工程师 Diana。")] + messages
    
    response = llm.invoke(messages)
    content = response.content.lower() if hasattr(response, "content") else ""
    
    if "bug" in content or "错误" in content or "issue" in content or "问题" in content:
        if "frontend" in content or "bob" in content or "前端" in content:
            return {"messages": [response], "active_agent": "bob", "iteration_count": iteration + 1}
        elif "backend" in content or "charlie" in content or "后端" in content:
            return {"messages": [response], "active_agent": "charlie", "iteration_count": iteration + 1}
        else:
            return {"messages": [response], "active_agent": "alice", "iteration_count": iteration + 1}
    
    if "approve" in content or "pass" in content or "通过" in content or "验收" in content:
        return {"messages": [response], "active_agent": "alice", "iteration_count": iteration + 1}
    
    return {"messages": [response], "active_agent": "diana", "iteration_count": iteration + 1}


def human_node(state: AgentState) -> dict:
    """Human-in-the-Loop 节点：暂停工作流等待人类输入。"""
    print("\n" + "="*60)
    print("🛑 工作流暂停：等待人类输入")
    print("="*60)
    
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and hasattr(last_message, "content"):
        print(f"\n代理请求：{last_message.content}")
    
    try:
        human_input = input("\n请输入您的回复（或输入 'continue' 继续）：")
    except (EOFError, KeyboardInterrupt):
        human_input = "continue"
    
    human_msg = HumanMessage(content=f"[Human Input] {human_input}")
    
    return {
        "messages": [human_msg],
        "active_agent": "alice",
        "pending_human": False,
        "human_input": human_input,
    }


# ───────────────────────────────────────────────────────
# 条件路由
# ───────────────────────────────────────────────────────

def route_agent(state: AgentState) -> Literal["alice", "bob", "charlie", "diana", "human", "end"]:
    """条件路由：根据 active_agent 状态字段决定下一个节点。"""
    active = state.get("active_agent", "alice")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 20)
    
    if iteration >= max_iter:
        print(f"⚠️ 达到最大迭代次数 {max_iter}，强制终止工作流。")
        return "end"
    
    if state.get("pending_human"):
        return "human"
    
    return active


# ───────────────────────────────────────────────────────
# 图构建
# ───────────────────────────────────────────────────────

def build_team_workflow() -> StateGraph:
    """
    构建四代理协作工作流图。
    
    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
    │  alice  │────▶│   bob   │     │ charlie │     │  diana  │     │  human  │
    │  (PM)   │◄────│(前端)    │     │(后端)    │     │  (QA)   │     │(人类)   │
    │  调度中心│     └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘
    │         │◄──────────┘◄──────────────┘◄──────────────┘◄──────────────┘
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │   END   │
    └─────────┘
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("alice", alice_node)
    workflow.add_node("bob", bob_node)
    workflow.add_node("charlie", charlie_node)
    workflow.add_node("diana", diana_node)
    workflow.add_node("human", human_node)
    
    workflow.set_entry_point("alice")
    
    for node_name in ["alice", "bob", "charlie", "diana", "human"]:
        workflow.add_conditional_edges(
            node_name,
            route_agent,
            {
                "alice": "alice",
                "bob": "bob",
                "charlie": "charlie",
                "diana": "diana",
                "human": "human",
                "end": END,
            },
        )
    
    return workflow.compile()


# 全局编译图
team_graph = build_team_workflow()
