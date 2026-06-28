"""MCP Server — FastMCP entrypoint exposing all tools for multi-agent collaboration."""

from typing import Optional, List

from mcp.server.fastmcp import FastMCP

from mcp_server.tools.github import (
    CreatePRInput, CreateBranchInput, CommitFileInput, ReadFileInput,
    ListCommitsInput, CreateIssueInput,
    github_create_pr, github_create_branch, github_commit_file,
    github_read_file, github_list_commits, github_create_issue,
)
from mcp_server.tools.slack import (
    SendMessageInput,
    slack_send_message, slack_list_channels, slack_get_channel_history,
    GetChannelHistoryInput,
)
from mcp_server.tools.ask_human import AskHumanInput, ask_human

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


# ── GitHub Tools ──

@mcp.tool()
async def github_create_pr_tool(
    repo_owner: str, repo_name: str, title: str, head_branch: str,
    base_branch: str = "main", body: Optional[str] = None, draft: bool = False,
) -> dict:
    """Create a GitHub Pull Request."""
    result = await github_create_pr(
        CreatePRInput(repo_owner=repo_owner, repo_name=repo_name, title=title,
                      head_branch=head_branch, base_branch=base_branch,
                      body=body, draft=draft)
    )
    return result.model_dump()


@mcp.tool()
async def github_create_branch_tool(
    repo_owner: str, repo_name: str, branch_name: str, from_branch: str = "main",
) -> dict:
    """Create a new branch in a GitHub repository."""
    result = await github_create_branch(
        CreateBranchInput(repo_owner=repo_owner, repo_name=repo_name,
                          branch_name=branch_name, from_branch=from_branch)
    )
    return result.model_dump()


@mcp.tool()
async def github_commit_file_tool(
    repo_owner: str, repo_name: str, branch: str, file_path: str,
    content: str, message: str,
) -> dict:
    """Commit a file to a GitHub repository branch."""
    result = await github_commit_file(
        CommitFileInput(repo_owner=repo_owner, repo_name=repo_name, branch=branch,
                        file_path=file_path, content=content, message=message)
    )
    return result.model_dump()


@mcp.tool()
async def github_read_file_tool(
    repo_owner: str, repo_name: str, file_path: str, branch: str = "main",
) -> dict:
    """Read a file from a GitHub repository."""
    result = await github_read_file(
        ReadFileInput(repo_owner=repo_owner, repo_name=repo_name,
                      file_path=file_path, branch=branch)
    )
    return result.model_dump()


@mcp.tool()
async def github_list_commits_tool(
    repo_owner: str, repo_name: str, branch: str = "main", per_page: int = 10,
) -> dict:
    """List recent commits on a GitHub branch."""
    result = await github_list_commits(
        ListCommitsInput(repo_owner=repo_owner, repo_name=repo_name,
                         branch=branch, per_page=per_page)
    )
    return result.model_dump()


@mcp.tool()
async def github_create_issue_tool(
    repo_owner: str, repo_name: str, title: str, body: str,
    labels: List[str] = [], assignees: List[str] = [],
) -> dict:
    """Create a GitHub Issue (e.g., for bug reports from QA)."""
    result = await github_create_issue(
        CreateIssueInput(repo_owner=repo_owner, repo_name=repo_name,
                         title=title, body=body, labels=labels, assignees=assignees)
    )
    return result.model_dump()


# ── Slack Tools ──

@mcp.tool()
async def slack_send_message_tool(
    channel: str, text: str, thread_ts: Optional[str] = None,
) -> dict:
    """Send a message to a Slack channel."""
    result = await slack_send_message(
        SendMessageInput(channel=channel, text=text, thread_ts=thread_ts)
    )
    return result.model_dump()


@mcp.tool()
async def slack_list_channels_tool() -> dict:
    """List all public Slack channels in the workspace."""
    result = await slack_list_channels()
    return result.model_dump()


@mcp.tool()
async def slack_get_channel_history_tool(
    channel: str, limit: int = 20,
) -> dict:
    """Get recent messages from a Slack channel."""
    result = await slack_get_channel_history(
        GetChannelHistoryInput(channel=channel, limit=limit)
    )
    return result.model_dump()


# ── Human-in-the-Loop Tool ──

@mcp.tool()
def ask_human_tool(
    question: str, context: Optional[str] = None, default_answer: Optional[str] = None,
) -> dict:
    """Pause agent execution and ask a human for input."""
    result = ask_human(
        AskHumanInput(question=question, context=context, default_answer=default_answer)
    )
    return result.model_dump()
