"""GitHub tools — production-hardened with shared HTTP client,
distributed rate limiting, and wall-clock retry budgets.
"""
from __future__ import annotations

import base64
import os
from typing import Optional, List

import httpx
from pydantic import BaseModel, Field

from shared.dependencies import (
    get_http_client,
    get_github_rate_limiter,
    RetryBudget,
    logger,
)

GITHUB_API_BASE = "https://api.github.com"
_GITHUB_TOKEN: str | None = None


def _get_token() -> str:
    global _GITHUB_TOKEN
    if _GITHUB_TOKEN is None:
        _GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
        if not _GITHUB_TOKEN:
            raise RuntimeError("GITHUB_TOKEN environment variable is required.")
    return _GITHUB_TOKEN


def _get_headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def _github_request(
    method: str,
    url: str,
    json: dict | None = None,
    params: dict | None = None,
) -> httpx.Response:
    """GitHub API request with distributed rate limiting and wall-clock retry budget."""
    client = get_http_client()
    headers = _get_headers()
    limiter = get_github_rate_limiter()

    rem, reset_at = await limiter.check()
    if rem <= 1 and reset_at > 0:
        import asyncio
        sleep_for = max(0, reset_at - int(__import__("time").time()) + 1)
        if sleep_for > 0:
            logger.warning("GitHub rate limit near exhaustion, sleeping %ds", sleep_for)
            await asyncio.sleep(min(sleep_for, 60))

    budget = RetryBudget(max_retries=3, max_total_seconds=120.0, base_delay=1.0)
    last_error: Exception | None = None

    for attempt in range(budget.max_retries):
        if not budget.should_retry(attempt):
            break

        try:
            resp = await client.request(method, url, headers=headers, json=json, params=params)
            try:
                new_rem = int(resp.headers.get("X-RateLimit-Remaining", rem))
                new_reset = int(resp.headers.get("X-RateLimit-Reset", 0))
                await limiter.update(new_rem, new_reset)
            except (ValueError, TypeError):
                pass

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                delay = budget.delay_for(attempt, int(retry_after) if retry_after else None)
                logger.warning("GitHub 429, retrying after %.1fs (attempt %d)", delay, attempt + 1)
                import asyncio
                await asyncio.sleep(delay)
                continue

            if resp.status_code >= 500:
                delay = budget.delay_for(attempt)
                logger.warning("GitHub %d, retrying after %.1fs", resp.status_code, delay)
                import asyncio
                await asyncio.sleep(delay)
                continue

            resp.raise_for_status()
            return resp

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_error = exc
            delay = budget.delay_for(attempt)
            import asyncio
            await asyncio.sleep(delay)
            continue
        except httpx.HTTPStatusError:
            raise

    raise RuntimeError(
        f"GitHub API request failed after retries or budget exhausted: {last_error}"
    )


class CreatePRInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the target repository")
    repo_name: str = Field(..., description="Name of the target repository")
    title: str = Field(..., description="PR title")
    head_branch: str = Field(..., description="Branch with changes")
    base_branch: str = Field(default="main", description="Branch to merge into")
    body: Optional[str] = Field(default=None, description="PR description")
    draft: bool = Field(default=False, description="Create as draft")


class CreatePROutput(BaseModel):
    pr_url: str
    pr_number: int
    state: str


class CreateBranchInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the repository")
    repo_name: str = Field(..., description="Name of the repository")
    branch_name: str = Field(..., description="New branch name")
    from_branch: str = Field(default="main", description="Source branch")


class CreateBranchOutput(BaseModel):
    ref: str
    sha: str
    url: str


class CommitFileInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the repository")
    repo_name: str = Field(..., description="Name of the repository")
    branch: str = Field(..., description="Target branch")
    file_path: str = Field(..., description="File path in the repository")
    content: str = Field(..., description="Raw file content (will be base64 encoded)")
    message: str = Field(..., description="Commit message")


class CommitFileOutput(BaseModel):
    sha: str
    commit_url: str


class ReadFileInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the repository")
    repo_name: str = Field(..., description="Name of the repository")
    file_path: str = Field(..., description="File path to read")
    branch: str = Field(default="main", description="Branch to read from")


class ReadFileOutput(BaseModel):
    content: str
    sha: str
    size: int


class CommitInfo(BaseModel):
    sha: str
    message: str
    author_name: str
    author_email: str
    date: str
    url: str


class ListCommitsInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the repository")
    repo_name: str = Field(..., description="Name of the repository")
    branch: str = Field(default="main", description="Branch to list commits from")
    per_page: int = Field(default=10, ge=1, le=100)


class ListCommitsOutput(BaseModel):
    commits: List[CommitInfo]


class CreateIssueInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the repository")
    repo_name: str = Field(..., description="Name of the repository")
    title: str = Field(..., description="Issue title")
    body: str = Field(..., description="Issue body")
    labels: List[str] = Field(default_factory=list)
    assignees: List[str] = Field(default_factory=list)


class CreateIssueOutput(BaseModel):
    issue_url: str
    issue_number: int
    state: str


async def github_create_pr(params: CreatePRInput) -> CreatePROutput:
    url = f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}/pulls"
    payload = {
        "title": params.title,
        "head": params.head_branch,
        "base": params.base_branch,
        "body": params.body,
        "draft": params.draft,
    }
    resp = await _github_request("POST", url, json=payload)
    data = resp.json()
    return CreatePROutput(pr_url=data["html_url"], pr_number=data["number"], state=data["state"])


async def github_create_branch(params: CreateBranchInput) -> CreateBranchOutput:
    base_url = f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}/git/refs/heads/{params.from_branch}"
    base_resp = await _github_request("GET", base_url)
    base_sha = base_resp.json()["object"]["sha"]
    create_url = f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}/git/refs"
    payload = {"ref": f"refs/heads/{params.branch_name}", "sha": base_sha}
    resp = await _github_request("POST", create_url, json=payload)
    data = resp.json()
    return CreateBranchOutput(ref=data["ref"], sha=data["object"]["sha"], url=data["url"])


async def github_commit_file(params: CommitFileInput) -> CommitFileOutput:
    url = f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}/contents/{params.file_path}"
    get_resp = await _github_request("GET", url, params={"ref": params.branch})
    existing_sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
    payload = {
        "message": params.message,
        "content": base64.b64encode(params.content.encode("utf-8")).decode("utf-8"),
        "branch": params.branch,
    }
    if existing_sha:
        payload["sha"] = existing_sha
    resp = await _github_request("PUT", url, json=payload)
    data = resp.json()
    return CommitFileOutput(sha=data["content"]["sha"], commit_url=data["commit"]["html_url"])


async def github_read_file(params: ReadFileInput) -> ReadFileOutput:
    url = f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}/contents/{params.file_path}"
    resp = await _github_request("GET", url, params={"ref": params.branch})
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return ReadFileOutput(content=content, sha=data["sha"], size=data["size"])


async def github_list_commits(params: ListCommitsInput) -> ListCommitsOutput:
    url = f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}/commits"
    resp = await _github_request("GET", url, params={"sha": params.branch, "per_page": params.per_page})
    data = resp.json()
    commits = [
        CommitInfo(
            sha=c["sha"],
            message=c["commit"]["message"],
            author_name=c["commit"]["author"]["name"],
            author_email=c["commit"]["author"]["email"],
            date=c["commit"]["author"]["date"],
            url=c["html_url"],
        )
        for c in data
    ]
    return ListCommitsOutput(commits=commits)


async def github_create_issue(params: CreateIssueInput) -> CreateIssueOutput:
    url = f"{GITHUB_API_BASE}/repos/{params.repo_owner}/{params.repo_name}/issues"
    payload = {
        "title": params.title,
        "body": params.body,
        "labels": params.labels,
        "assignees": params.assignees,
    }
    resp = await _github_request("POST", url, json=payload)
    data = resp.json()
    return CreateIssueOutput(issue_url=data["html_url"], issue_number=data["number"], state=data["state"])
