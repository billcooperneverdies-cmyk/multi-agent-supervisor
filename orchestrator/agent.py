"""Agent CLI — 交互式四代理团队协作运行器。"""

import asyncio
import uuid

from langgraph.types import Command

from orchestrator.state_machine import (
    make_run_config,
    new_task_state,
    normalize_content,
    team_graph,
)


async def run_team(input_message: str, thread_ts: str | None = None) -> None:
    """运行完整的四代理团队协作工作流（含 HITL 中断恢复）。"""
    task_id = str(uuid.uuid4())[:8]
    initial_state = new_task_state(
        input_message, task_id, 20, slack_thread_ts=thread_ts
    )
    config = make_run_config(task_id, 20)

    print(f"\n🚀 启动团队协作：{input_message}")
    print("=" * 60)

    final_state = await team_graph.ainvoke(initial_state, config=config)

    # HITL：图在 human 节点 interrupt()，这里通过 Command(resume=...) 恢复。
    while "__interrupt__" in final_state:
        payload = final_state["__interrupt__"][0]
        request = getattr(payload, "value", payload)
        if isinstance(request, dict):
            request = request.get("request", "")
        print(f"\n🛑 代理请求人类输入：{normalize_content(request)}")
        try:
            answer = input("您（输入回复继续）: ")
        except (EOFError, KeyboardInterrupt):
            answer = "continue"
            print("\n(stdin 不可用，自动回复 'continue')")
        final_state = await team_graph.ainvoke(Command(resume=answer), config=config)

    print("\n" + "=" * 60)
    print("✅ 工作流完成")
    print(f"总迭代次数：{final_state['iteration_count']}")
    print(f"最终活跃代理：{final_state['active_agent']}")

    print("\n📋 完整对话记录：")
    for i, msg in enumerate(final_state.get("messages", [])):
        role = getattr(msg, "type", "unknown")
        content = normalize_content(getattr(msg, "content", str(msg)))[:200]
        print(f"  [{i}] {role}: {content}...")


async def main() -> int:
    """交互式 CLI 运行器。

    BUG-036: stdin EOF（容器 / Ctrl-D）干净退出；单个任务失败不终结会话；
    返回明确的退出码。
    """
    print("🤖 AI 软件开发团队 (Alice · Bob · Charlie · Diana)")
    print("输入需求描述，团队将自动协作完成开发。")
    print("输入 'exit' 退出。\n")

    while True:
        try:
            user_input = input("您: ")
        except EOFError:
            print("\n👋 输入流已关闭 (EOF)，退出。")
            return 0
        except KeyboardInterrupt:
            print("\n👋 收到中断信号 (Ctrl-C)，退出。")
            return 130

        if user_input.strip().lower() in ("exit", "quit"):
            print("👋 再见。")
            return 0
        if not user_input.strip():
            continue

        try:
            await run_team(user_input)
        except Exception as exc:
            print(f"\n❌ 任务执行失败：{type(exc).__name__}: {exc}")
            print("请检查 LiteLLM 代理 / MCP 服务是否可用，然后重试。")
        print()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
