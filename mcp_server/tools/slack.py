"""Extended Slack MCP Tools — 支持多代理频道通信。

- SLACK_BOT_TOKEN is read lazily at call time (BUG-019).
- Every outbound client uses the explicit module-level HTTP_TIMEOUT (BUG-035).
- All failures map to ToolError with a uniform {ok, error, code} shape (BUG-043).
"""

import os
import re

import httpx
from pydantic import BaseModel, Field

from mcp_server.tools.errors import HTTP_TIMEOUT, ToolError, map_api_errors

SLACK_API_BASE = os.environ.get("SLACK_API_BASE", "https://slack.com/api")

# BUG-042: cap the cursor loop at a sane number of pages.
# BUG-043: per-process cache of channel name -> channel ID.
MAX_LIST_PAGES = 10
_channel_cache: dict = {}

# Slack conversation IDs: C... (channel), D... (DM), G... (group/private).
_CHANNEL_ID_RE = re.compile(r"^[CDG][A-Z0-9]+$")


def _headers() -> dict:
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        raise ToolError(
            "SLACK_BOT_TOKEN is not configured; Slack tools are unavailable",
            code="not_configured",
        )
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }


def _check_ok(data: dict) -> dict:
    """Slack returns HTTP 200 with ok:false for API-level errors."""
    if not isinstance(data, dict) or not data.get("ok"):
        error_msg = (data or {}).get("error", "unknown_error") if isinstance(data, dict) \
            else "unknown_error"
        raise ToolError(f"Slack API returned error: {error_msg}", code="slack_error")
    return data


class SendMessageInput(BaseModel):
    channel: str = Field(..., description="Channel ID or name (e.g., #alerts, C123456)")
    text: str = Field(..., description="Message text in plain text or mrkdwn")
    thread_ts: str | None = Field(
        default=None, description="Thread timestamp to reply in a thread"
    )


class SendMessageOutput(BaseModel):
    ok: bool
    channel: str
    ts: str
    thread_ts: str | None = None


@map_api_errors("Slack")
async def slack_send_message(params: SendMessageInput) -> SendMessageOutput:
    """Post a message to a Slack channel using chat.postMessage."""
    url = f"{SLACK_API_BASE}/chat.postMessage"
    channel_id = await _resolve_channel_id(params.channel)
    payload = {
        "channel": channel_id,
        "text": params.text,
    }
    if params.thread_ts:
        payload["thread_ts"] = params.thread_ts

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
        resp.raise_for_status()
        data = _check_ok(resp.json())

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
    channels: list[ChannelInfo]


async def _fetch_channel_pages(client: httpx.AsyncClient,
                               max_pages: int = MAX_LIST_PAGES) -> list[dict]:
    """Follow conversations.list cursors (BUG-042), capped at ``max_pages``."""
    url = f"{SLACK_API_BASE}/conversations.list"
    channels: list[dict] = []
    cursor: str | None = None
    for _ in range(max_pages):
        params: dict = {"types": "public_channel", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        resp = await client.get(url, headers=_headers(), params=params)
        resp.raise_for_status()
        data = _check_ok(resp.json())
        channels.extend(data.get("channels") or [])
        meta = data.get("metadata")
        if not isinstance(meta, dict):
            meta = data.get("response_metadata") or {}
        cursor = meta.get("next_cursor") or ""
        if not cursor:
            break
    return channels


@map_api_errors("Slack")
async def slack_list_channels(max_pages: int = MAX_LIST_PAGES) -> ListChannelsOutput:
    """List public Slack channels, following cursor pagination."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        raw_channels = await _fetch_channel_pages(client, max_pages=max_pages)

    channels = [
        ChannelInfo(
            id=c["id"],
            name=c["name"],
            is_private=c.get("is_private", False),
            num_members=c.get("num_members", 0),
        )
        for c in raw_channels
    ]
    return ListChannelsOutput(channels=channels)


@map_api_errors("Slack")
async def _resolve_channel_id(channel: str) -> str:
    """Accept C.../D.../G... IDs directly; resolve #name/name via cached list (BUG-043)."""
    if _CHANNEL_ID_RE.match(channel or ""):
        return channel
    name = (channel or "").lstrip("#").strip()
    if not name:
        raise ToolError(
            f"channel must be a Slack ID (C.../D.../G...) or a channel name "
            f"(got {channel!r})",
            code="invalid_input", status=400,
        )
    if name in _channel_cache:
        return _channel_cache[name]
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        raw_channels = await _fetch_channel_pages(client)
    for c in raw_channels:
        if isinstance(c, dict) and c.get("name") and c.get("id"):
            _channel_cache.setdefault(c["name"], c["id"])
    if name not in _channel_cache:
        raise ToolError(
            f"unknown Slack channel name: {channel!r} (not found in workspace)",
            code="channel_not_found", status=404,
        )
    return _channel_cache[name]


class MessageInfo(BaseModel):
    ts: str
    text: str
    user: str
    thread_ts: str | None = None


class GetChannelHistoryInput(BaseModel):
    channel: str = Field(..., description="Channel ID or name")
    limit: int = Field(default=20, ge=1, le=100)


class GetChannelHistoryOutput(BaseModel):
    messages: list[MessageInfo]


@map_api_errors("Slack")
async def slack_get_channel_history(params: GetChannelHistoryInput) -> GetChannelHistoryOutput:
    """Get recent messages from a Slack channel."""
    url = f"{SLACK_API_BASE}/conversations.history"
    channel_id = await _resolve_channel_id(params.channel)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.get(
            url,
            headers=_headers(),
            params={"channel": channel_id, "limit": params.limit},
        )
        resp.raise_for_status()
        data = _check_ok(resp.json())

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
