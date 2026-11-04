"""Tests for Slack MCP tools: BUG-019/035/042/043.

Cursor pagination, channel-name -> ID resolution with per-process cache,
uniform structured errors, lazy token handling. All HTTP mocked with respx.
"""

import asyncio
import json

import httpx
import pytest
import respx

from mcp_server.tools.errors import ToolError
from mcp_server.tools import slack as sl


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    sl._channel_cache.clear()


LIST_URL = "https://slack.com/api/conversations.list"
POST_URL = "https://slack.com/api/chat.postMessage"
HIST_URL = "https://slack.com/api/conversations.history"


def _page(channels, next_cursor=""):
    return {"ok": True, "channels": channels,
            "metadata": {"next_cursor": next_cursor},
            "response_metadata": {"next_cursor": next_cursor}}


# ── BUG-019: lazy token ───────────────────────────────────────────────


def test_missing_slack_token_structured_error(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    with pytest.raises(ToolError) as exc:
        run(sl.slack_send_message(sl.SendMessageInput(channel="C1", text="hi")))
    assert exc.value.code == "not_configured"


# ── BUG-042: cursor pagination ────────────────────────────────────────


@respx.mock
def test_list_channels_follows_next_cursor():
    pages = [
        _page([{"id": "C1", "name": "a"}], next_cursor="cur2"),
        _page([{"id": "C2", "name": "b"}], next_cursor=""),
    ]
    respx.get(LIST_URL).mock(side_effect=[httpx.Response(200, json=p) for p in pages])
    out = run(sl.slack_list_channels())
    assert [c.id for c in out.channels] == ["C1", "C2"]
    assert len(respx.calls) == 2
    assert respx.calls[1].request.url.params["cursor"] == "cur2"


@respx.mock
def test_list_channels_caps_at_max_pages():
    respx.get(LIST_URL).mock(return_value=httpx.Response(
        200, json=_page([{"id": "C1", "name": "a"}], next_cursor="more")))
    out = run(sl.slack_list_channels())
    assert len(respx.calls) == sl.MAX_LIST_PAGES == 10
    assert len(out.channels) == 10


@respx.mock
def test_list_channels_response_metadata_variant():
    page = {"ok": True, "channels": [{"id": "C1", "name": "a"}],
            "response_metadata": {"next_cursor": "c2"}}
    page2 = {"ok": True, "channels": [{"id": "C2", "name": "b"}],
             "response_metadata": {"next_cursor": ""}}
    respx.get(LIST_URL).mock(
        side_effect=[httpx.Response(200, json=page), httpx.Response(200, json=page2)])
    out = run(sl.slack_list_channels())
    assert [c.id for c in out.channels] == ["C1", "C2"]


# ── BUG-043: name -> ID resolution + uniform errors ──────────────────


@respx.mock
def test_channel_name_resolved_to_id():
    respx.get(LIST_URL).mock(return_value=httpx.Response(
        200, json=_page([{"id": "C123", "name": "general"}])))
    respx.post(POST_URL).mock(return_value=httpx.Response(
        200, json={"ok": True, "channel": "C123", "ts": "1.0"}))
    out = run(sl.slack_send_message(sl.SendMessageInput(channel="#general", text="hi")))
    assert out.channel == "C123"
    post_body = json.loads(respx.calls[1].request.content)
    assert post_body["channel"] == "C123"


@respx.mock
def test_channel_name_resolution_cached_per_process():
    respx.get(LIST_URL).mock(return_value=httpx.Response(
        200, json=_page([{"id": "C123", "name": "general"}])))
    respx.post(POST_URL).mock(return_value=httpx.Response(
        200, json={"ok": True, "channel": "C123", "ts": "1.0"}))
    run(sl.slack_send_message(sl.SendMessageInput(channel="#general", text="hi")))
    run(sl.slack_send_message(sl.SendMessageInput(channel="general", text="hi2")))
    list_calls = [c for c in respx.calls if c.request.url.path.endswith("conversations.list")]
    assert len(list_calls) == 1  # second send used the cache


@respx.mock
def test_channel_id_passes_through_without_list_call():
    respx.post(POST_URL).mock(return_value=httpx.Response(
        200, json={"ok": True, "channel": "C999", "ts": "1.0"}))
    out = run(sl.slack_send_message(sl.SendMessageInput(channel="C999", text="hi")))
    assert out.channel == "C999"
    assert len(respx.calls) == 1


@respx.mock
def test_unknown_channel_name_structured_error():
    respx.get(LIST_URL).mock(return_value=httpx.Response(
        200, json=_page([{"id": "C123", "name": "general"}])))
    with pytest.raises(ToolError) as exc:
        run(sl.slack_send_message(sl.SendMessageInput(channel="#nope", text="hi")))
    assert exc.value.code == "channel_not_found"


@respx.mock
def test_history_resolves_channel_name():
    respx.get(LIST_URL).mock(return_value=httpx.Response(
        200, json=_page([{"id": "C123", "name": "general"}])))
    respx.get(HIST_URL).mock(return_value=httpx.Response(
        200, json={"ok": True, "messages": [{"ts": "1", "text": "hi", "user": "U1"}]}))
    out = run(sl.slack_get_channel_history(
        sl.GetChannelHistoryInput(channel="#general", limit=5)))
    assert out.messages[0].text == "hi"
    assert respx.calls[1].request.url.params["channel"] == "C123"


@respx.mock
def test_slack_ok_false_maps_to_uniform_error():
    respx.post(POST_URL).mock(return_value=httpx.Response(
        200, json={"ok": False, "error": "not_in_channel"}))
    with pytest.raises(ToolError) as exc:
        run(sl.slack_send_message(sl.SendMessageInput(channel="C1", text="hi")))
    assert exc.value.code == "slack_error"
    assert "not_in_channel" in exc.value.message


@respx.mock
def test_slack_http_error_maps_to_uniform_error():
    respx.get(HIST_URL).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(ToolError) as exc:
        run(sl.slack_get_channel_history(
            sl.GetChannelHistoryInput(channel="C1", limit=5)))
    assert exc.value.code == "http_error"
    assert exc.value.status == 500


@respx.mock
def test_server_wrapper_returns_uniform_error_dict():
    respx.post(POST_URL).mock(return_value=httpx.Response(
        200, json={"ok": False, "error": "not_in_channel"}))
    import mcp_server.server as srv

    result = run(srv.slack_send_message_tool(channel="C1", text="hi"))
    assert result == {"ok": False, "error": result["error"], "code": "slack_error"}
    assert "not_in_channel" in result["error"]


def test_slack_timeout_config():
    assert sl.HTTP_TIMEOUT.connect == 5.0
    assert sl.HTTP_TIMEOUT.read == 30.0
