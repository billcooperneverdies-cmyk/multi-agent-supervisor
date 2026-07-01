"""Slack tools — production-hardened with shared HTTP client
and distributed rate limiting.
"""
from __future__ import annotations

import os
from typing import Optional, List

import httpx
from pydantic import BaseModel, Field

from shared.dependencies import (
    get_http_client,
    get_slack_rate_limiter,
    RetryBudget,
    logger,
)

SLACK_API_BASE = "https://slack.com/api"
_SLACK_TOKEN: str | None = None


def _get_token() -> str:
    global _SLACK_TOKEN
    if _SLACK_TOKEN is None:
        _SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
        if not _SLACK_TOKEN:
            raise RuntimeError("SLACK_BOT_TOKEN environment variable is required.")
    return _SLACK_TOKEN


def _get_headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }


async def _slack_request(method: str, url: str, json: dict | None = None, params: dict | None = None) -> dict:
    client = get_http_client()
    headers = _get_headers()
    limiter = get_slack_rate_limiter()

    budget = RetryBudget(max_retries=3, max_total_seconds=60.0, base_delay=1.0)
    last_error: Exception | None = None

    for attempt in range(budget.max_retries):
        if not budget.should_retry(attempt):
            break

        try:
            resp = await client.request(method, url, headers=headers, json=json, params=params)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("ok"):
                error = data.get("error", "unknown_error")
                if error == "rate_limited":
                    retry_after = int(resp.headers.get("Retry-After", 1))
                    delay = budget.delay_for(attempt, retry_after)
                    logger.warning("Slack rate limited, retrying after %.1fs", delay)
                    import asyncio
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(f"Slack API error: {error}")

            return data

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_error = exc
            delay = budget.delay_for(attempt)
            import asyncio
            await asyncio.sleep(delay)
            continue

    raise RuntimeError(f"Slack API request failed: {last_error}")


class SendMessageInput(BaseModel):
    channel: str = Field(..., description="Channel ID or name (e.g., #general)")
    text: str = Field(..., description="Message text")
    thread_ts: Optional[str] = Field(default=None, description="Thread timestamp for replies")


class SendMessageOutput(BaseModel):
    ok: bool
    channel: str
    ts: str


class ChannelInfo(BaseModel):
    id: str
    name: str


class ListChannelsOutput(BaseModel):
    channels: List[ChannelInfo]


class MessageInfo(BaseModel):
    ts: str
    text: str
    user: str


class GetChannelHistoryInput(BaseModel):
    channel: str = Field(..., description="Channel ID")
    limit: int = Field(default=20, ge=1, le=200)


class GetChannelHistoryOutput(BaseModel):
    messages: List[MessageInfo]


async def slack_send_message(params: SendMessageInput) -> SendMessageOutput:
    url = f"{SLACK_API_BASE}/chat.postMessage"
    payload = {"channel": params.channel, "text": params.text}
    if params.thread_ts:
        payload["thread_ts"] = params.thread_ts
    data = await _slack_request("POST", url, json=payload)
    return SendMessageOutput(ok=data["ok"], channel=data["channel"], ts=data["ts"])


async def slack_list_channels() -> ListChannelsOutput:
    url = f"{SLACK_API_BASE}/conversations.list"
    data = await _slack_request("GET", url, params={"types": "public_channel"})
    channels = [ChannelInfo(id=c["id"], name=c["name"]) for c in data.get("channels", [])]
    return ListChannelsOutput(channels=channels)


async def slack_get_channel_history(params: GetChannelHistoryInput) -> GetChannelHistoryOutput:
    url = f"{SLACK_API_BASE}/conversations.history"
    data = await _slack_request("GET", url, params={"channel": params.channel, "limit": params.limit})
    messages = [
        MessageInfo(ts=m["ts"], text=m.get("text", ""), user=m.get("user", ""))
        for m in data.get("messages", [])
    ]
    return GetChannelHistoryOutput(messages=messages)
