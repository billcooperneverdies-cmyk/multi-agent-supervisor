"""Human-in-the-Loop Tool — 当代理需要人类确认时暂停并请求输入。"""

from pydantic import BaseModel, Field
from typing import Optional


class AskHumanInput(BaseModel):
    """请求人类输入的 schema。"""

    question: str = Field(..., description="向人类提出的问题")
    context: Optional[str] = Field(
        default=None, description="额外的上下文信息，帮助人类理解问题"
    )
    default_answer: Optional[str] = Field(
        default=None, description="默认回答（如果用户不提供输入）"
    )


class AskHumanOutput(BaseModel):
    """人类回答的 schema。"""

    answer: str
    confirmed: bool = Field(default=True, description="用户是否确认了操作")


def ask_human(params: AskHumanInput) -> AskHumanOutput:
    """
    暂停代理执行，请求人类输入。

    在 LangGraph 中，此工具应该被实现为中断节点（interrupt node），
    当代理调用此工具时，工作流暂停，等待人类通过 UI 或 CLI 提供输入。
    """
    print(f"\n{'='*60}")
    print(f"🤖 代理需要人类确认：")
    print(f"{'='*60}")
    print(f"\n{params.question}")
    if params.context:
        print(f"\n上下文：{params.context}")
    print(f"\n{'='*60}")

    try:
        answer = input(f"\n请输入您的回答 (默认: {params.default_answer or '否'}): ")
    except (EOFError, KeyboardInterrupt):
        answer = params.default_answer or "否"

    confirmed = answer.lower() in ("yes", "y", "是", "确认", "ok", "true")

    return AskHumanOutput(answer=answer, confirmed=confirmed)
