"""Human-in-the-Loop Tool — 当代理需要人类确认时暂停并请求输入。

BUG-026: this tool runs inside a stdio-transport MCP server, where stdout is the
JSON-RPC byte stream and stdin is the inbound message stream. It therefore
NEVER prints to stdout and NEVER reads stdin:

- If the connected MCP client supports elicitation (mcp >= 1.28 FastMCP
  ``Context.elicit``), the question is asked through the protocol and the
  answer is returned with ``status="answered"``.
- Otherwise the tool returns a structured ``status="pending_human"`` payload
  that the client/orchestrator can answer out-of-band (e.g. via a LangGraph
  interrupt + resume), without corrupting stdio framing.

All diagnostics go to stderr via logging.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("mcp_server.tools.ask_human")


class AskHumanInput(BaseModel):
    """请求人类输入的 schema。"""

    question: str = Field(..., description="向人类提出的问题")
    context: str | None = Field(
        default=None, description="额外的上下文信息，帮助人类理解问题"
    )
    default_answer: str | None = Field(
        default=None, description="默认回答（如果用户不提供输入）"
    )


class HumanAnswerSchema(BaseModel):
    """Elicitation schema (primitive types only, per the MCP elicitation spec)."""

    answer: str = Field(..., description="The human's answer to the question")


class AskHumanOutput(BaseModel):
    """Structured result: an answered question or a pending_human handoff."""

    status: str = Field(..., description="'answered' or 'pending_human'")
    question: str
    answer: str | None = None
    confirmed: bool = Field(default=False, description="用户是否确认了操作")
    default_answer: str | None = None


_CONFIRM_WORDS = ("yes", "y", "是", "确认", "ok", "true")


def _is_confirmed(answer: str) -> bool:
    return answer.strip().lower() in _CONFIRM_WORDS


async def ask_human(params: AskHumanInput, ctx: Any | None = None) -> AskHumanOutput:
    """Ask a human without touching stdout/stdin.

    ``ctx`` is the FastMCP Context injected by the server wrapper; when present
    (and the client supports elicitation) the question is asked in-protocol.
    """
    logger.info("ask_human requested: %s", params.question)

    if ctx is not None:
        message = params.question
        if params.context:
            message += f"\n\nContext: {params.context}"
        if params.default_answer is not None:
            message += f"\n(default: {params.default_answer})"
        try:
            result = await ctx.elicit(message=message, schema=HumanAnswerSchema)
        except Exception:  # noqa: BLE001 - client may not support elicitation
            logger.info("elicitation unavailable; returning pending_human", exc_info=True)
        else:
            if result.action == "accept" and result.data is not None:
                answer = result.data.answer
                return AskHumanOutput(
                    status="answered",
                    question=params.question,
                    answer=answer,
                    confirmed=_is_confirmed(answer),
                    default_answer=params.default_answer,
                )
            logger.info("elicitation %s; returning pending_human", result.action)

    return AskHumanOutput(
        status="pending_human",
        question=params.question,
        answer=None,
        confirmed=False,
        default_answer=params.default_answer,
    )
