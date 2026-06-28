"""Extended Slack MCP Tools — 支持多代理频道通信。"""

import os
from typing import Optional, List

import httpx
from pydantic import BaseModel, Field

SLACK_API_BASE = "https://slack.com/api"
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
if not SLACK_BOT_TOKEN:
    raise RuntimeError("SLACK_BOT_TOKEN environment variable is required.")

DEFAULT_HEADERS = {
    "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
    "Content-Type": "application/json; charset=utf-8",
}


class SendMessageInput(BaseModel):
    channel: str = Field(..., description="Channel ID or name (e.g., #alerts, C123456)")
    text: str = Field(..., description="Message text in plain text or mrkdwn")
    thread_ts: Optional[str] = Field(
        default=None, description="Thread timestamp to reply in a thread"
    )


class SendMessageOutput(BaseModel):
    ok: bool
    channel: str
    ts: str
    thread_ts: Optional[str] = None


async def slack_send_message(params: SendMessageInput) -> SendMessageOutput:
    """Post a message to a Slack channel using chat.postMessage."""
    url = f"{SLACK_API_BASE}/chat.postMessage"
    payload = {
        "channel": params.channel,
        "text": params.text,
    }
    if params.thread_ts:
        payload["thread_ts"] = params.thread_ts

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=DEFAULT_HEADERS, json=payload)
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


class ChannelInfo(BaseModel):
    id: str
    name: str
    is_private: bool
    num_members: int


class ListChannelsOutput(BaseModel):
    channels: List[ChannelInfo]


async def slack_list_channels() -> ListChannelsOutput:
    """List all public Slack channels in the workspace."""
    url = f"{SLACK_API_BASE}/conversations.list"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=DEFAULT_HEADERS, params={"types": "public_channel"})
        resp.raise_for_status()
        data = resp.json()

    if not data.get("ok"):
        error_msg = data.get("error", "unknown_error")
        raise RuntimeError(f"Slack API returned error: {error_msg}")

    channels = [
        ChannelInfo(
            id=c["id"],
            name=c["name"],
            is_private=c.get("is_private", False),
            num_members=c.get("num_members", 0),
        )
        for c in data.get("channels", [])
    ]
    return ListChannelsOutput(channels=channels)


class MessageInfo(BaseModel):
    ts: str
    text: str
    user: str
    thread_ts: Optional[str] = None


class GetChannelHistoryInput(BaseModel):
    channel: str = Field(..., description="Channel ID or name")
    limit: int = Field(default=20, ge=1, le=100)


class GetChannelHistoryOutput(BaseModel):
    messages: List[MessageInfo]


async def slack_get_channel_history(params: GetChannelHistoryInput) -> GetChannelHistoryOutput:
    """Get recent messages from a Slack channel."""
    url = f"{SLACK_API_BASE}/conversations.history"
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            url,
            headers=DEFAULT_HEADERS,
            params={"channel": params.channel, "limit": params.limit},
        )
        resp.raise_for_status()
        data = resp.json()

    if not data.get("ok"):
        error_msg = data.get("error", "unknown_error")
        raise RuntimeError(f"Slack API returned error: {error_msg}")

    messages = [
        MessageInfo(
            ts=m["ts"],
            text=m.get("text", ""),
            user=m.get("user", ""),
            thread_ts=m.get("thread_ts"),
        )
        for m in data.get("messages", [])
    ]
    return GetChannelHistoryOutput(messages=messages)
