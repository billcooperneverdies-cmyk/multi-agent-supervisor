"""Extended GitHub MCP Tools — 支持多代理协作开发工作流。"""

import os
from typing import Optional, List

import httpx
from pydantic import BaseModel, Field

GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN environment variable is required.")

DEFAULT_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


class CreatePRInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the target repository")
    repo_name: str = Field(..., description="Name of the target repository")
    title: str = Field(..., description="Title of the pull request")
    head_branch: str = Field(..., description="Branch containing changes")
    base_branch: str = Field(default="main", description="Branch to merge into")
    body: Optional[str] = Field(default=None, description="Markdown body for the PR")
    draft: bool = Field(default=False, description="Create as draft PR")


class CreatePROutput(BaseModel):
    pr_url: str
    pr_number: int
    state: str


async def github_create_pr(params: CreatePRInput) -> CreatePROutput:
    """Create a pull request via the GitHub REST API v3."""
    url = f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}/pulls"
    payload = {
        "title": params.title,
        "head": params.head_branch,
        "base": params.base_branch,
        "body": params.body or "",
        "draft": params.draft,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=DEFAULT_HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return CreatePROutput(
        pr_url=data["html_url"],
        pr_number=data["number"],
        state=data["state"],
    )


class CreateBranchInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the target repository")
    repo_name: str = Field(..., description="Name of the target repository")
    branch_name: str = Field(..., description="Name of the new branch")
    from_branch: str = Field(default="main", description="Base branch to branch from")


class CreateBranchOutput(BaseModel):
    ref: str
    sha: str
    url: str


async def github_create_branch(params: CreateBranchInput) -> CreateBranchOutput:
    """Create a new branch from an existing branch."""
    base_url = (
        f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}"
        f"/git/refs/heads/{params.from_branch}"
    )
    async with httpx.AsyncClient() as client:
        base_resp = await client.get(base_url, headers=DEFAULT_HEADERS)
        base_resp.raise_for_status()
        base_sha = base_resp.json()["object"]["sha"]

    create_url = (
        f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}"
        f"/git/refs"
    )
    payload = {
        "ref": f"refs/heads/{params.branch_name}",
        "sha": base_sha,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(create_url, headers=DEFAULT_HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return CreateBranchOutput(ref=data["ref"], sha=data["object"]["sha"], url=data["url"])


class CommitFileInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the target repository")
    repo_name: str = Field(..., description="Name of the target repository")
    branch: str = Field(..., description="Branch to commit to")
    file_path: str = Field(..., description="Path in the repository (e.g., frontend/App.tsx)")
    content: str = Field(..., description="File content as plain text")
    message: str = Field(..., description="Commit message")


class CommitFileOutput(BaseModel):
    sha: str
    commit_url: str


async def github_commit_file(params: CommitFileInput) -> CommitFileOutput:
    """Create or update a file in a repository."""
    import base64

    url = (
        f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}"
        f"/contents/{params.file_path}"
    )
    encoded_content = base64.b64encode(params.content.encode("utf-8")).decode("utf-8")

    async with httpx.AsyncClient() as client:
        get_resp = await client.get(
            url, headers=DEFAULT_HEADERS, params={"ref": params.branch}
        )
        existing_sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

    payload = {
        "message": params.message,
        "content": encoded_content,
        "branch": params.branch,
    }
    if existing_sha:
        payload["sha"] = existing_sha

    async with httpx.AsyncClient() as client:
        resp = await client.put(url, headers=DEFAULT_HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return CommitFileOutput(
        sha=data["content"]["sha"],
        commit_url=data["commit"]["html_url"],
    )


class ReadFileInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the target repository")
    repo_name: str = Field(..., description="Name of the target repository")
    file_path: str = Field(..., description="Path in the repository")
    branch: str = Field(default="main", description="Branch to read from")


class ReadFileOutput(BaseModel):
    content: str
    sha: str
    size: int


async def github_read_file(params: ReadFileInput) -> ReadFileOutput:
    """Read a file from a repository."""
    import base64

    url = (
        f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}"
        f"/contents/{params.file_path}"
    )
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=DEFAULT_HEADERS, params={"ref": params.branch})
        resp.raise_for_status()
        data = resp.json()

    content = base64.b64decode(data["content"]).decode("utf-8")
    return ReadFileOutput(content=content, sha=data["sha"], size=data["size"])


class ListCommitsInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the target repository")
    repo_name: str = Field(..., description="Name of the target repository")
    branch: str = Field(default="main", description="Branch to list commits from")
    per_page: int = Field(default=10, ge=1, le=100)


class CommitInfo(BaseModel):
    sha: str
    message: str
    author: str
    date: str
    url: str


class ListCommitsOutput(BaseModel):
    commits: List[CommitInfo]


async def github_list_commits(params: ListCommitsInput) -> ListCommitsOutput:
    """List recent commits on a branch."""
    url = f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}/commits"
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            url,
            headers=DEFAULT_HEADERS,
            params={"sha": params.branch, "per_page": params.per_page},
        )
        resp.raise_for_status()
        data = resp.json()

    commits = [
        CommitInfo(
            sha=c["sha"],
            message=c["commit"]["message"],
            author=c["commit"]["author"]["name"],
            date=c["commit"]["author"]["date"],
            url=c["html_url"],
        )
        for c in data
    ]
    return ListCommitsOutput(commits=commits)


class CreateIssueInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the target repository")
    repo_name: str = Field(..., description="Name of the target repository")
    title: str = Field(..., description="Issue title")
    body: str = Field(..., description="Issue body (Markdown)")
    labels: List[str] = Field(default_factory=list, description="Labels to apply")
    assignees: List[str] = Field(default_factory=list, description="Assignees")


class CreateIssueOutput(BaseModel):
    issue_url: str
    issue_number: int
    state: str


async def github_create_issue(params: CreateIssueInput) -> CreateIssueOutput:
    """Create a GitHub issue (e.g., for bug reports from QA)."""
    url = f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}/issues"
    payload = {
        "title": params.title,
        "body": params.body,
        "labels": params.labels,
        "assignees": params.assignees,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=DEFAULT_HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return CreateIssueOutput(
        issue_url=data["html_url"],
        issue_number=data["number"],
        state=data["state"],
    )
