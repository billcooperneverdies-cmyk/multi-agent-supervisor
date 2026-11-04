"""BUG-036: CLI lifecycle — EOF on stdin exits cleanly, per-task errors are
survivable, exit codes are sane. No tracebacks, no restart loops."""

import asyncio

import pytest


def test_cli_eof_exits_cleanly(monkeypatch, capsys):
    """Closed stdin (Ctrl-D / container) -> clean message + exit code 0."""
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    from orchestrator.agent import main

    rc = asyncio.run(main())
    assert rc == 0
    out = capsys.readouterr().out
    assert "Traceback" not in out


def test_cli_exit_command(monkeypatch):
    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    from orchestrator.agent import main

    assert asyncio.run(main()) == 0


def test_cli_task_error_does_not_kill_session(monkeypatch, capsys, install_model):
    """A failing run is reported and the loop continues; EOF then exits 0."""

    class Boom(Exception):
        pass

    async def boom(*args, **kwargs):
        raise Boom("litellm proxy unreachable")

    monkeypatch.setattr("orchestrator.agent.run_team", boom)
    inputs = iter(["do something"])
    calls = {"n": 0}

    def fake_input(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return next(inputs)
        raise EOFError()

    monkeypatch.setattr("builtins.input", fake_input)
    from orchestrator.agent import main

    rc = asyncio.run(main())
    assert rc == 0
    out = capsys.readouterr().out
    assert "litellm proxy unreachable" in out
    assert "Traceback" not in out


def test_cli_keyboard_interrupt_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    from orchestrator.agent import main

    rc = asyncio.run(main())
    assert rc == 130
    assert "Traceback" not in capsys.readouterr().out
