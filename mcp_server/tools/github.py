"""Extended GitHub MCP Tools — 支持多代理协作开发工作流。

Security/resilience notes:
- GITHUB_TOKEN is read lazily at call time (BUG-019): importing this module and
  starting the server never requires the token; calling a GitHub tool without it
  yields a structured ``not_configured`` ToolError.
- Every outbound client uses the explicit module-level HTTP_TIMEOUT (BUG-035).
- Response parsing is guarded; unexpected shapes become ToolError instead of
  bare KeyError/JSONDecodeError (BUG-040).
"""

import base64
import os
import re
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from mcp_server.tools.errors import HTTP_TIMEOUT, ToolError, map_api_errors

GITHUB_API_BASE = os.environ.get("GITHUB_API_BASE", "https://api.github.com")

# BUG-020: strict allowlist for URL path segments interpolated into GitHub API
# URLs (tool inputs are LLM-driven, i.e. prompt-injection-influenced).
_NAME_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_repo_segment(value: str, field: str) -> str:
    """Validate a repo owner/name segment: no slashes, dot-segments, or empties."""
    if not value or value in (".", "..") or not _NAME_SEGMENT_RE.match(value):
        raise ToolError(
            f"{field} must match ^[A-Za-z0-9_.-]+$ and must not be '.'/'..' "
            f"(got {value!r})",
            code="invalid_input", status=400,
        )
    return value


def _encode_path_segments(path: str, field: str) -> str:
    """Validate + percent-encode a slash-separated path for safe URL interpolation.

    Rejects empty/leading-slash paths and any empty, '.', '..', '?' or '#'
    segment; intended sub-directory slashes are preserved, everything else is
    percent-encoded with quote(safe='') so the value can never reshape the URL.
    """
    if not path or path.startswith("/"):
        raise ToolError(
            f"{field} must be a non-empty relative path without a leading '/' "
            f"(got {path!r})",
            code="invalid_input", status=400,
        )
    segments = path.split("/")
    for seg in segments:
        if seg in ("", ".", "..") or "?" in seg or "#" in seg:
            raise ToolError(
                f"{field} contains illegal segment {seg!r}: empty, '.', '..', "
                f"'?' and '#' segments are not allowed",
                code="invalid_input", status=400,
            )
    return "/".join(quote(seg, safe="") for seg in segments)


def _validate_branch_name(value: str, field: str = "branch_name") -> str:
    """Branch names may contain '/' but never '..', '?', '#', or a leading '/'."""
    if not value or value.startswith("/") or ".." in value or "?" in value or "#" in value:
        raise ToolError(
            f"{field} must be a valid git branch name without '..', '?', '#' "
            f"or a leading '/' (got {value!r})",
            code="invalid_input", status=400,
        )
    return value


def _repo_base(owner: str, name: str) -> str:
    _validate_repo_segment(owner, "repo_owner")
    _validate_repo_segment(name, "repo_name")
    return f"{GITHUB_API_BASE}/repos/{quote(owner, safe='')}/{quote(name, safe='')}"


def _headers() -> dict[str, str]:
    """Build auth headers at call time; structured error if unconfigured."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ToolError(
            "GITHUB_TOKEN is not configured; GitHub tools are unavailable",
            code="not_configured",
        )
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


class CreatePRInput(BaseModel):
    repo_owner: str = Field(..., description="Owner of the target repository")
    repo_name: str = Field(..., description="Name of the target repository")
    title: str = Field(..., description="Title of the pull request")
    head_branch: str = Field(..., description="Branch containing changes")
    base_branch: str = Field(default="main", description="Branch to merge into")
    body: str | None = Field(default=None, description="Markdown body for the PR")
    draft: bool = Field(default=False, description="Create as draft PR")


class CreatePROutput(BaseModel):
    pr_url: str
    pr_number: int
    state: str


@map_api_errors("GitHub")
async def github_create_pr(params: CreatePRInput) -> CreatePROutput:
    """Create a pull request via the GitHub REST API v3."""
    url = f"{_repo_base(params.repo_owner, params.repo_name)}/pulls"
    _validate_branch_name(params.head_branch, "head_branch")
    _validate_branch_name(params.base_branch, "base_branch")
    payload = {
        "title": params.title,
        "head": params.head_branch,
        "base": params.base_branch,
        "body": params.body or "",
        "draft": params.draft,
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
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


@map_api_errors("GitHub")
async def github_create_branch(params: CreateBranchInput) -> CreateBranchOutput:
    """Create a new branch from an existing branch."""
    base_url = (
        f"{_repo_base(params.repo_owner, params.repo_name)}"
        f"/git/refs/heads/{_encode_path_segments(params.from_branch, 'from_branch')}"
    )
    _validate_branch_name(params.branch_name)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        base_resp = await client.get(base_url, headers=_headers())
        base_resp.raise_for_status()
        base_sha = base_resp.json()["object"]["sha"]

    create_url = (
        f"{_repo_base(params.repo_owner, params.repo_name)}"
        f"/git/refs"
    )
    payload = {
        "ref": f"refs/heads/{params.branch_name}",
        "sha": base_sha,
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(create_url, headers=_headers(), json=payload)
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


@map_api_errors("GitHub")
async def github_commit_file(params: CommitFileInput) -> CommitFileOutput:
    """Create or update a file in a repository."""
    url = (
        f"{_repo_base(params.repo_owner, params.repo_name)}"
        f"/contents/{_encode_path_segments(params.file_path, 'file_path')}"
    )
    encoded_content = base64.b64encode(params.content.encode("utf-8")).decode("utf-8")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        get_resp = await client.get(
            url, headers=_headers(), params={"ref": params.branch}
        )
        if get_resp.status_code == 200:
            try:
                existing_sha = get_resp.json().get("sha")
            except ValueError:
                existing_sha = None
        else:
            existing_sha = None

    payload = {
        "message": params.message,
        "content": encoded_content,
        "branch": params.branch,
    }
    if existing_sha:
        payload["sha"] = existing_sha

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.put(url, headers=_headers(), json=payload)
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


@map_api_errors("GitHub")
async def github_read_file(params: ReadFileInput) -> ReadFileOutput:
    """Read a file from a repository."""
    url = (
        f"{_repo_base(params.repo_owner, params.repo_name)}"
        f"/contents/{_encode_path_segments(params.file_path, 'file_path')}"
    )
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.get(url, headers=_headers(), params={"ref": params.branch})
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
    commits: list[CommitInfo]


@map_api_errors("GitHub")
async def github_list_commits(params: ListCommitsInput) -> ListCommitsOutput:
    """List recent commits on a branch."""
    url = f"{_repo_base(params.repo_owner, params.repo_name)}/commits"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.get(
            url,
            headers=_headers(),
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
    labels: list[str] = Field(default_factory=list, description="Labels to apply")
    assignees: list[str] = Field(default_factory=list, description="Assignees")


class CreateIssueOutput(BaseModel):
    issue_url: str
    issue_number: int
    state: str


@map_api_errors("GitHub")
async def github_create_issue(params: CreateIssueInput) -> CreateIssueOutput:
    """Create a GitHub issue (e.g., for bug reports from QA)."""
    url = f"{_repo_base(params.repo_owner, params.repo_name)}/issues"
    payload = {
        "title": params.title,
        "body": params.body,
        "labels": params.labels,
        "assignees": params.assignees,
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()

    return CreateIssueOutput(
        issue_url=data["html_url"],
        issue_number=data["number"],
        state=data["state"],
    )
