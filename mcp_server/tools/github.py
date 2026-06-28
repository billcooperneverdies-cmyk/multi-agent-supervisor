"""GitHub MCP Tool — Create Pull Request."""

import os
from typing import Optional

import httpx
from pydantic import BaseModel, Field

GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN environment variable is required.")


# ───────────────────────────────────────────────────────
# Input / Output Schemas
# ───────────────────────────────────────────────────────
class CreatePRInput(BaseModel):
    """Schema for github_create_pr tool input."""

    repo_owner: str = Field(..., description="Owner of the target repository")
    repo_name: str = Field(..., description="Name of the target repository")
    title: str = Field(..., description="Title of the pull request")
    head_branch: str = Field(..., description="Branch containing changes")
    base_branch: str = Field(default="main", description="Branch to merge into")
    body: Optional[str] = Field(default=None, description="Markdown body for the PR")
    draft: bool = Field(default=False, description="Create as draft PR")


class CreatePROutput(BaseModel):
    """Schema for github_create_pr tool output."""

    pr_url: str
    pr_number: int
    state: str


# ───────────────────────────────────────────────────────
# Tool Implementation
# ───────────────────────────────────────────────────────
async def github_create_pr(params: CreatePRInput) -> CreatePROutput:
    """Create a pull request via the GitHub REST API v3."""
    url = f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}/pulls"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "title": params.title,
        "head": params.head_branch,
        "base": params.base_branch,
        "body": params.body or "",
        "draft": params.draft,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return CreatePROutput(
        pr_url=data["html_url"],
        pr_number=data["number"],
        state=data["state"],
    )
