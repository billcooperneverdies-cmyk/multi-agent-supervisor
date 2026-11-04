"""MCP Server — FastMCP entrypoint exposing all tools for multi-agent collaboration.

Tool wrappers convert ToolError into a structured {"ok": False, error, code}
dict (BUG-019/040/043) so failures never traceback into the JSON-RPC protocol.
"""

import logging

from mcp.server.fastmcp import Context, FastMCP

from mcp_server.tools.ask_human import AskHumanInput, ask_human
from mcp_server.tools.errors import ToolError
from mcp_server.tools.github import (
    CommitFileInput,
    CreateBranchInput,
    CreateIssueInput,
    CreatePRInput,
    ListCommitsInput,
    ReadFileInput,
    github_commit_file,
    github_create_branch,
    github_create_issue,
    github_create_pr,
    github_list_commits,
    github_read_file,
)
from mcp_server.tools.slack import (
    GetChannelHistoryInput,
    SendMessageInput,
    slack_get_channel_history,
    slack_list_channels,
    slack_send_message,
)

logger = logging.getLogger("mcp_server.server")

mcp = FastMCP(
    name="multi-agent-tools",
    instructions=(
        "MCP Server exposing GitHub, Slack, and Human-in-the-Loop tools for "
        "an AI software development team (Alice PM, Bob Frontend, Charlie Backend, Diana QA).\n"
        "GitHub tools: create_branch, commit_file, create_pr, read_file, list_commits, create_issue\n"
        "Slack tools: send_message, list_channels, get_channel_history\n"
        "Human tools: ask_human (for clarification requests)"
    ),
)


async def _guard(coro) -> dict:
    """Run a tool coroutine; map failures to a structured error dict."""
    try:
        result = await coro
        return result.model_dump()
    except ToolError as exc:
        return exc.as_dict()
    except Exception as exc:  # noqa: BLE001 - last line of defense for the protocol
        logger.exception("unexpected tool failure")
        return {"ok": False, "error": f"internal_error: {exc}", "code": "internal_error"}


# ── GitHub Tools ──

@mcp.tool()
async def github_create_pr_tool(
    repo_owner: str, repo_name: str, title: str, head_branch: str,
    base_branch: str = "main", body: str | None = None, draft: bool = False,
) -> dict:
    """Create a GitHub Pull Request."""
    return await _guard(github_create_pr(
        CreatePRInput(repo_owner=repo_owner, repo_name=repo_name, title=title,
                      head_branch=head_branch, base_branch=base_branch,
                      body=body, draft=draft)
    ))


@mcp.tool()
async def github_create_branch_tool(
    repo_owner: str, repo_name: str, branch_name: str, from_branch: str = "main",
) -> dict:
    """Create a new branch in a GitHub repository."""
    return await _guard(github_create_branch(
        CreateBranchInput(repo_owner=repo_owner, repo_name=repo_name,
                          branch_name=branch_name, from_branch=from_branch)
    ))


@mcp.tool()
async def github_commit_file_tool(
    repo_owner: str, repo_name: str, branch: str, file_path: str,
    content: str, message: str,
) -> dict:
    """Commit a file to a GitHub repository branch."""
    return await _guard(github_commit_file(
        CommitFileInput(repo_owner=repo_owner, repo_name=repo_name, branch=branch,
                        file_path=file_path, content=content, message=message)
    ))


@mcp.tool()
async def github_read_file_tool(
    repo_owner: str, repo_name: str, file_path: str, branch: str = "main",
) -> dict:
    """Read a file from a GitHub repository."""
    return await _guard(github_read_file(
        ReadFileInput(repo_owner=repo_owner, repo_name=repo_name,
                      file_path=file_path, branch=branch)
    ))


@mcp.tool()
async def github_list_commits_tool(
    repo_owner: str, repo_name: str, branch: str = "main", per_page: int = 10,
) -> dict:
    """List recent commits on a GitHub branch."""
    return await _guard(github_list_commits(
        ListCommitsInput(repo_owner=repo_owner, repo_name=repo_name,
                         branch=branch, per_page=per_page)
    ))


@mcp.tool()
async def github_create_issue_tool(
    repo_owner: str, repo_name: str, title: str, body: str,
    labels: list[str] | None = None, assignees: list[str] | None = None,
) -> dict:
    """Create a GitHub Issue (e.g., for bug reports from QA)."""
    return await _guard(github_create_issue(
        CreateIssueInput(repo_owner=repo_owner, repo_name=repo_name,
                         title=title, body=body,
                         labels=labels or [], assignees=assignees or [])
    ))


# ── Slack Tools ──

@mcp.tool()
async def slack_send_message_tool(
    channel: str, text: str, thread_ts: str | None = None,
) -> dict:
    """Send a message to a Slack channel (ID or #name)."""
    return await _guard(slack_send_message(
        SendMessageInput(channel=channel, text=text, thread_ts=thread_ts)
    ))


@mcp.tool()
async def slack_list_channels_tool() -> dict:
    """List all public Slack channels in the workspace."""
    return await _guard(slack_list_channels())


@mcp.tool()
async def slack_get_channel_history_tool(
    channel: str, limit: int = 20,
) -> dict:
    """Get recent messages from a Slack channel (ID or #name)."""
    return await _guard(slack_get_channel_history(
        GetChannelHistoryInput(channel=channel, limit=limit)
    ))


# ── Human-in-the-Loop Tool ──

@mcp.tool()
async def ask_human_tool(
    question: str, ctx: Context,
    context: str | None = None, default_answer: str | None = None,
) -> dict:
    """Ask a human for input.

    Uses MCP elicitation when the client supports it; otherwise returns a
    structured ``status="pending_human"`` payload for out-of-band answering.
    Never writes to stdout or reads stdin (stdio framing integrity).
    """
    result = await ask_human(
        AskHumanInput(question=question, context=context, default_answer=default_answer),
        ctx=ctx,
    )
    return result.model_dump()
