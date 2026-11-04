"""Server-level tests: BUG-019 (no-token start) and BUG-026 (stdio framing).

- Server must import and start with ZERO tokens set.
- tools/list must return all 10 tools.
- ask_human must not write to stdout or read stdin: every stdout byte of
  the stdio server must parse as JSON-RPC, and the call must complete
  without stdin input.
"""

import asyncio
import json
import os
import select
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _clean_env():
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    # Propagate venv hints so the same interpreter finds its site-packages.
    for key in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        if key in os.environ:
            env[key] = os.environ[key]
    env.pop("GITHUB_TOKEN", None)
    env.pop("SLACK_BOT_TOKEN", None)
    return env


# ── BUG-019 ───────────────────────────────────────────────────────────


def test_server_imports_with_zero_tokens():
    proc = subprocess.run(
        [sys.executable, "-c", "import mcp_server.server; print('IMPORT_OK')"],
        env=_clean_env(), capture_output=True, text=True, timeout=120, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "IMPORT_OK" in proc.stdout
    assert "RuntimeError" not in proc.stderr


def test_server_module_starts_with_empty_env_and_no_crash_loop():
    """env -i style smoke: server must not die with RuntimeError traceback.

    With stdin closed it may exit cleanly (EOF on transport) or keep
    serving; it must NOT crash with the import-time RuntimeError.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "mcp_server.main"],
        env=_clean_env(), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=120, cwd=REPO_ROOT,
    )
    assert "RuntimeError" not in proc.stderr
    assert "Traceback" not in proc.stderr or "RuntimeError" not in proc.stderr


# ── BUG-026 ───────────────────────────────────────────────────────────


def _rpc_initialize(msg_id=1):
    return {
        "jsonrpc": "2.0", "id": msg_id, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "framing-test", "version": "0"},
        },
    }


def _read_json_line(proc, timeout=30):
    """Read one stdout line; assert it parses as a JSON-RPC message."""
    fd = proc.stdout.fileno()
    ready, _, _ = select.select([fd], [], [], timeout)
    assert ready, "timed out waiting for server stdout (server blocked?)"
    line = proc.stdout.readline()
    assert line, "server closed stdout unexpectedly"
    msg = json.loads(line.decode("utf-8"))  # raises if non-JSON bytes emitted
    assert msg.get("jsonrpc") == "2.0"
    return msg


def test_ask_human_over_stdio_keeps_framing_and_needs_no_stdin():
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.main"],
        env=_clean_env(), cwd=REPO_ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        def send(msg):
            proc.stdin.write(json.dumps(msg).encode() + b"\n")
            proc.stdin.flush()

        send(_rpc_initialize(1))
        resp = _read_json_line(proc)
        assert resp["id"] == 1 and "result" in resp

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        # tools/list: all 10 tools served even with zero tokens (BUG-019).
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        resp = _read_json_line(proc)
        while resp.get("id") != 2:
            resp = _read_json_line(proc)
        tool_names = sorted(t["name"] for t in resp["result"]["tools"])
        assert len(tool_names) == 10, tool_names

        # tools/call ask_human: must complete without any stdin answer and
        # must not corrupt stdout framing (old code print()ed a banner and
        # blocked on input()).
        send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "ask_human_tool",
                       "arguments": {"question": "Deploy to prod?",
                                     "default_answer": "no"}},
        })
        resp = _read_json_line(proc, timeout=30)
        while resp.get("id") != 3:
            if resp.get("method") == "elicitation/create":
                # Client does not support elicitation: decline in-protocol.
                # The tool must fall back to a pending_human payload.
                send({"jsonrpc": "2.0", "id": resp["id"],
                      "error": {"code": -32601,
                                "message": "elicitation not supported"}})
            resp = _read_json_line(proc, timeout=30)
        assert "result" in resp, resp
        content = resp["result"]["content"]
        payload = json.loads(content[0]["text"])
        assert payload["status"] == "pending_human"
        assert payload["question"] == "Deploy to prod?"
        assert resp["result"].get("isError") in (False, None)
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_ask_human_tool_function_is_stdio_clean(capsys):
    """Direct call: no stdout writes, no stdin reads, structured result."""
    import mcp_server.server as srv

    result = asyncio.run(srv.ask_human_tool(question="Proceed?", ctx=None))
    out, _ = capsys.readouterr()
    assert out == ""
    assert result["status"] == "pending_human"
    assert result["question"] == "Proceed?"
