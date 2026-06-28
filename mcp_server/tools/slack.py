"""Slack MCP Tool — Send Message."""

import os
from typing import Optional

import httpx
from pydantic import BaseModel, Field

SLACK_API_BASE = "https://slack.com/api"
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
if not SLACK_BOT_TOKEN:
    raise RuntimeError("SLACK_BOT_TOKEN environment variable is required.")


# ───────────────────────────────────────────────────────
# Input / Output Schemas
# ───────────────────────────────────────────────────────
class SendMessageInput(BaseModel):
    """Schema for slack_send_message tool input."""

    channel: str = Field(..., description="Channel ID or name (e.g., #alerts, C123456)")
    text: str = Field(..., description="Message text in plain text or mrkdwn")
    thread_ts: Optional[str] = Field(
        default=None, description="Thread timestamp to reply in a thread"
    )


class SendMessageOutput(BaseModel):
    """Schema for slack_send_message tool output."""

    ok: bool
    channel: str
    ts: str
    thread_ts: Optional[str] = None


# ───────────────────────────────────────────────────────
# Tool Implementation
# ───────────────────────────────────────────────────────
async def slack_send_message(params: SendMessageInput) -> SendMessageOutput:
    """Post a message to a Slack channel using chat.postMessage."""
    url = f"{SLACK_API_BASE}/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {
        "channel": params.channel,
        "text": params.text,
    }
    if params.thread_ts:
        payload["thread_ts"] = params.thread_ts

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    if not data.get("ok"):
        error_msg = data.get("error", "unknown_error")
        raise RuntimeError(f"Slack API returned error: {error_msg}")

    return SendMessageOutput(
        ok=data["ok"],
        channel=data["channel"],
        ts=data["ts"],
        thread_ts=data.get("thread_ts"),
    )
