"""Agent CLI — 交互式四代理团队协作运行器。"""

import asyncio
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.state_machine import AgentState, team_graph

LITELLM_API_BASE = os.environ.get("LITELLM_API_BASE", "http://localhost:4000")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-litellm-master-key")


def create_llm(model: str = "gpt-4o") -> ChatOpenAI:
    """初始化通过 LiteLLM 代理的 LLM。"""
    return ChatOpenAI(
        model=model,
        api_key=LITELLM_API_KEY,
        base_url=f"{LITELLM_API_BASE}/v1",
        temperature=0.2,
    )


async def run_team(input_message: str, thread_ts: str | None = None) -> None:
    """运行完整的四代理团队协作工作流。"""
    initial_state: AgentState = {
        "messages": [HumanMessage(content=input_message)],
        "active_agent": "alice",
        "task_id": None,
        "task_description": input_message,
        "project_board": None,
        "slack_thread_ts": thread_ts,
        "slack_channel": "#general",
        "repo_owner": None,
        "repo_name": None,
        "current_branch": None,
        "frontend_branch": None,
        "backend_branch": None,
        "pr_urls": [],
        "issue_urls": [],
        "tool_calls": None,
        "tool_results": None,
        "iteration_count": 0,
        "max_iterations": 20,
        "human_input": None,
        "pending_human": False,
    }

    print(f"\n🚀 启动团队协作：{input_message}")
    print("=" * 60)
    
    final_state = await team_graph.ainvoke(initial_state)
    
    print("\n" + "=" * 60)
    print("✅ 工作流完成")
    print(f"总迭代次数：{final_state['iteration_count']}")
    print(f"最终活跃代理：{final_state['active_agent']}")
    
    print("\n📋 完整对话记录：")
    for i, msg in enumerate(final_state.get("messages", [])):
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))[:200]
        print(f"  [{i}] {role}: {content}...")


async def main() -> None:
    """交互式 CLI 运行器。"""
    print("🤖 AI 软件开发团队 (Alice · Bob · Charlie · Diana)")
    print("输入需求描述，团队将自动协作完成开发。")
    print("输入 'exit' 退出。\n")

    while True:
        user_input = input("您: ")
        if user_input.strip().lower() in ("exit", "quit"):
            break
        await run_team(user_input)
        print()


if __name__ == "__main__":
    asyncio.run(main())
