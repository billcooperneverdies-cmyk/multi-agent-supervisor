"""Tests for GitHub MCP tools: BUG-019/020/035/040.

Covers lazy token handling, URL path-injection validation, explicit
httpx timeouts, and guarded response parsing. All HTTP is mocked with
respx; no network access.
"""

import asyncio
import inspect

import httpx
import pytest
import respx

from mcp_server.tools.errors import ToolError
from mcp_server.tools import github as gh


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")


# ── BUG-019: lazy token handling ──────────────────────────────────────


def test_missing_token_raises_structured_tool_error(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(ToolError) as exc:
        run(gh.github_read_file(gh.ReadFileInput(
            repo_owner="octo", repo_name="demo", file_path="README.md")))
    assert exc.value.code == "not_configured"


def test_missing_token_makes_no_http_call(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with respx.mock(assert_all_called=False):
        with pytest.raises(ToolError):
            run(gh.github_create_issue(gh.CreateIssueInput(
                repo_owner="octo", repo_name="demo", title="t", body="b")))
        assert len(respx.calls) == 0


def test_server_wrapper_returns_structured_error_without_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    import mcp_server.server as srv

    result = run(srv.github_read_file_tool(
        repo_owner="octo", repo_name="demo", file_path="README.md"))
    assert result["ok"] is False
    assert result["code"] == "not_configured"
    assert "GITHUB_TOKEN" in result["error"]


# ── BUG-020: URL path injection validation ────────────────────────────


@pytest.mark.parametrize("owner", ["a/b", "..", "", "a?b", "a#b", "a b", "a/b/c"])
def test_repo_owner_injection_rejected(owner):
    with pytest.raises(ToolError) as exc:
        run(gh.github_read_file(gh.ReadFileInput(
            repo_owner=owner, repo_name="demo", file_path="README.md")))
    assert exc.value.code == "invalid_input"
    assert "repo_owner" in exc.value.message


@pytest.mark.parametrize("name", ["a/b", "..", ".", "", "x?y"])
def test_repo_name_injection_rejected(name):
    with pytest.raises(ToolError) as exc:
        run(gh.github_read_file(gh.ReadFileInput(
            repo_owner="octo", repo_name=name, file_path="README.md")))
    assert exc.value.code == "invalid_input"
    assert "repo_name" in exc.value.message


@pytest.mark.parametrize("path", [
    "../../admin/settings", "../x", "/abs/path", "a//b", "a/./b",
    "x?ref=evil", "x#frag", "",
])
def test_file_path_injection_rejected(path):
    with pytest.raises(ToolError) as exc:
        run(gh.github_commit_file(gh.CommitFileInput(
            repo_owner="octo", repo_name="demo", branch="main",
            file_path=path, content="x", message="m")))
    assert exc.value.code == "invalid_input"
    assert "file_path" in exc.value.message


@respx.mock(assert_all_called=False)
def test_file_path_dotdot_never_reaches_http():
    with pytest.raises(ToolError):
        run(gh.github_read_file(gh.ReadFileInput(
            repo_owner="octo", repo_name="demo", file_path="../../x")))
    assert len(respx.calls) == 0


@respx.mock(assert_all_called=False)
def test_owner_slash_never_reaches_http():
    with pytest.raises(ToolError):
        run(gh.github_read_file(gh.ReadFileInput(
            repo_owner="a/b", repo_name="demo", file_path="x")))
    assert len(respx.calls) == 0


@respx.mock
def test_file_path_segments_url_encoded_and_scoped():
    route = respx.get(url__regex=r"https://api\.github\.com/repos/octo/demo/contents/.*")
    route.mock(side_effect=httpx.Response(404, json={"message": "nf"}))
    with pytest.raises(ToolError):
        run(gh.github_read_file(gh.ReadFileInput(
            repo_owner="octo", repo_name="demo", file_path="dir/my file+1.txt")))
    req = respx.calls[0].request
    assert req.url.raw_path.split(b"?")[0] == \
        b"/repos/octo/demo/contents/dir/my%20file%2B1.txt"


@respx.mock
def test_branch_with_slash_allowed_but_encoded():
    respx.get(
        "https://api.github.com/repos/octo/demo/git/refs/heads/feat/x"
    ).mock(return_value=httpx.Response(404, json={"message": "nf"}))
    with pytest.raises(ToolError):  # 404 -> http_error, but URL shape is what we assert
        run(gh.github_create_branch(gh.CreateBranchInput(
            repo_owner="octo", repo_name="demo", branch_name="b", from_branch="feat/x")))
    assert respx.calls[0].request.url.path == "/repos/octo/demo/git/refs/heads/feat/x"


@pytest.mark.parametrize("branch", ["..", "x?y", "a#b", "/lead", ""])
def test_from_branch_injection_rejected(branch):
    with pytest.raises(ToolError) as exc:
        run(gh.github_create_branch(gh.CreateBranchInput(
            repo_owner="octo", repo_name="demo", branch_name="b", from_branch=branch)))
    assert exc.value.code == "invalid_input"


# ── BUG-035: explicit timeouts ────────────────────────────────────────


def test_timeout_config_defaults():
    assert gh.HTTP_TIMEOUT.connect == 5.0
    assert gh.HTTP_TIMEOUT.read == 30.0


@respx.mock
def test_timeout_passed_to_every_client(monkeypatch):
    captured = []
    real_client = httpx.AsyncClient

    class CapturingClient(real_client):
        def __init__(self, *args, **kwargs):
            captured.append(kwargs.get("timeout"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", CapturingClient)
    respx.post("https://api.github.com/repos/octo/demo/issues").mock(
        return_value=httpx.Response(
            201, json={"html_url": "u", "number": 1, "state": "open"}))
    run(gh.github_create_issue(gh.CreateIssueInput(
        repo_owner="octo", repo_name="demo", title="t", body="b")))
    assert captured and all(t is gh.HTTP_TIMEOUT for t in captured)


# ── BUG-040: guarded response parsing ─────────────────────────────────


@respx.mock
def test_create_branch_unexpected_ref_payload_is_tool_error():
    respx.get(
        "https://api.github.com/repos/octo/demo/git/refs/heads/main"
    ).mock(return_value=httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ToolError) as exc:
        run(gh.github_create_branch(gh.CreateBranchInput(
            repo_owner="octo", repo_name="demo", branch_name="b")))
    assert exc.value.code == "bad_response"


@respx.mock
def test_non_json_response_is_tool_error():
    respx.get(url__regex=r"https://api\.github\.com/repos/octo/demo/contents/.*").mock(
        return_value=httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(ToolError) as exc:
        run(gh.github_read_file(gh.ReadFileInput(
            repo_owner="octo", repo_name="demo", file_path="README.md")))
    assert exc.value.code == "bad_response"


@respx.mock
def test_http_error_is_structured_tool_error():
    respx.post("https://api.github.com/repos/octo/demo/pulls").mock(
        return_value=httpx.Response(422, json={"message": "Validation Failed"}))
    with pytest.raises(ToolError) as exc:
        run(gh.github_create_pr(gh.CreatePRInput(
            repo_owner="octo", repo_name="demo", title="t", head_branch="h")))
    assert exc.value.code == "http_error"
    assert exc.value.status == 422


@respx.mock
def test_list_commits_missing_author_fields_do_not_traceback():
    respx.get("https://api.github.com/repos/octo/demo/commits").mock(
        return_value=httpx.Response(200, json=[{"sha": "abc", "commit": {}}]))
    with pytest.raises(ToolError) as exc:
        run(gh.github_list_commits(gh.ListCommitsInput(
            repo_owner="octo", repo_name="demo")))
    assert exc.value.code == "bad_response"


@respx.mock
def test_timeout_maps_to_tool_error(monkeypatch):
    respx.get("https://api.github.com/repos/octo/demo/commits").mock(
        side_effect=httpx.ReadTimeout("boom"))
    with pytest.raises(ToolError) as exc:
        run(gh.github_list_commits(gh.ListCommitsInput(
            repo_owner="octo", repo_name="demo")))
    assert exc.value.code == "timeout"


# ── BUG-041: mutable defaults (server signature) ──────────────────────


def test_no_mutable_defaults_in_create_issue_tool():
    import mcp_server.server as srv

    sig = inspect.signature(srv.github_create_issue_tool)
    assert sig.parameters["labels"].default is None
    assert sig.parameters["assignees"].default is None


@respx.mock
def test_successive_issue_calls_do_not_share_list_identity():
    payloads = []
    respx.post("https://api.github.com/repos/octo/demo/issues").mock(
        side_effect=lambda req: (
            payloads.append(__import__("json").loads(req.content)),
            httpx.Response(201, json={"html_url": "u", "number": 1, "state": "open"}),
        )[1]
    )
    import mcp_server.server as srv

    run(srv.github_create_issue_tool(repo_owner="octo", repo_name="demo",
                                     title="t", body="b"))
    run(srv.github_create_issue_tool(repo_owner="octo", repo_name="demo",
                                     title="t", body="b"))
    assert len(payloads) == 2
    assert payloads[0]["labels"] == [] and payloads[1]["labels"] == []
    assert payloads[0]["labels"] is not payloads[1]["labels"]
