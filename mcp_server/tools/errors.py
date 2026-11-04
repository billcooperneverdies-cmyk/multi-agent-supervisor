"""Shared structured errors, HTTP timeout config, and error mapping for MCP tools.

Design notes:
- ``ToolError`` is the single error type tools raise; FastMCP wrapper functions in
  ``mcp_server.server`` convert it to a structured ``{"ok": False, ...}`` dict so a
  tool failure never tracebacks into the JSON-RPC protocol layer.
- ``HTTP_TIMEOUT`` is explicit (no reliance on httpx library defaults) and
  env-overridable: ``MCP_HTTP_TIMEOUT`` (read/write/pool, default 30s) and
  ``MCP_HTTP_CONNECT_TIMEOUT`` (connect, default 5s).
- ``map_api_errors`` normalizes httpx/parse failures into ``ToolError``.
"""

import functools
import json
import os
from collections.abc import Callable
from typing import Any

import httpx


class ToolError(Exception):
    """A structured, protocol-safe tool failure."""

    def __init__(self, message: str, code: str = "tool_error",
                 status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status

    def as_dict(self) -> dict:
        return {"ok": False, "error": self.message, "code": self.code}


def _timeout_from_env() -> httpx.Timeout:
    read = float(os.environ.get("MCP_HTTP_TIMEOUT", "30"))
    connect = float(os.environ.get("MCP_HTTP_CONNECT_TIMEOUT", "5"))
    return httpx.Timeout(read, connect=connect)


# Module-level timeout applied to EVERY outbound httpx client in the tools.
HTTP_TIMEOUT = _timeout_from_env()


def map_api_errors(service: str) -> Callable:
    """Decorator: normalize httpx/parse failures of an async tool into ToolError."""

    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except ToolError:
                raise
            except httpx.TimeoutException as exc:
                raise ToolError(f"{service} API request timed out", code="timeout") from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                detail = ""
                if exc.response is not None:
                    try:
                        detail = exc.response.text[:200]
                    except Exception:  # noqa: BLE001 - best-effort detail only
                        detail = ""
                raise ToolError(
                    f"{service} API HTTP {status}: {detail}".strip(),
                    code="http_error", status=status,
                ) from exc
            except httpx.RequestError as exc:
                raise ToolError(f"{service} API request failed: {exc}",
                                code="http_error") from exc
            except json.JSONDecodeError as exc:
                raise ToolError(f"invalid JSON in {service} API response",
                                code="bad_response") from exc
            except (KeyError, TypeError, AttributeError) as exc:
                raise ToolError(f"unexpected {service} API response shape ({exc!r})",
                                code="bad_response") from exc
            except ValueError as exc:  # e.g. bad base64, pydantic validation
                raise ToolError(f"unexpected {service} API response value ({exc!r})",
                                code="bad_response") from exc

        return wrapper

    return decorate
