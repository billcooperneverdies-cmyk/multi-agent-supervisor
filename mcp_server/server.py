"""MCP Server — FastMCP entrypoint exposing GitHub & Slack tools."""

from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.tools.github import CreatePRInput, github_create_pr
from mcp_server.tools.slack import SendMessageInput, slack_send_message

# ───────────────────────────────────────────────────────
# FastMCP Server Initialization
# ───────────────────────────────────────────────────────
mcp = FastMCP(
    name="multi-agent-tools",
    instructions=(
        "MCP Server exposing GitHub and Slack tools for an AI agent orchestration layer. "
        "Use github_create_pr to open pull requests and slack_send_message to notify channels."
    ),
)


@mcp.tool()
async def github_create_pr_tool(
    repo_owner: str,
    repo_name: str,
    title: str,
    head_branch: str,
    base_branch: str = "main",
    body: Optional[str] = None,
    draft: bool = False,
) -> dict:
    """
    Create a GitHub Pull Request.

    Args:
        repo_owner: Owner of the target repository (e.g., "octocat")
        repo_name: Name of the target repository (e.g., "hello-world")
        title: Title of the pull request
        head_branch: Branch containing the changes to merge
        base_branch: Branch to merge into (default: "main")
        body: Markdown body for the PR (optional)
        draft: Create as a draft PR (default: False)

    Returns:
        dict with pr_url, pr_number, and state.
    """
    result = await github_create_pr(
        CreatePRInput(
            repo_owner=repo_owner,
            repo_name=repo_name,
            title=title,
            head_branch=head_branch,
            base_branch=base_branch,
            body=body,
            draft=draft,
        )
    )
    return result.model_dump()


@mcp.tool()
async def slack_send_message_tool(
    channel: str,
    text: str,
    thread_ts: Optional[str] = None,
) -> dict:
    """
    Send a message to a Slack channel.

    Args:
        channel: Channel ID or name (e.g., "#alerts", "C1234567890")
        text: Message text (plain text or mrkdwn)
        thread_ts: Timestamp of a parent message to reply in a thread (optional)

    Returns:
        dict with ok, channel, ts, and thread_ts.
    """
    result = await slack_send_message(
        SendMessageInput(channel=channel, text=text, thread_ts=thread_ts)
    )
    return result.model_dump()
