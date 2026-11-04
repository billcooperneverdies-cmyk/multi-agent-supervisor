"""BUG-001 import smoke test.

The repo previously shipped a local ``langgraph/`` package that shadowed the
real PyPI ``langgraph`` distribution, so ``from langgraph.graph import ...``
resolved into the local package and died with
``ModuleNotFoundError: No module named 'langgraph.graph'``.

The local package is renamed to ``orchestrator/``. These tests assert:
  (a) the documented entry-point modules import under their new name, and
  (b) ``import langgraph`` now resolves to the real PyPI package
      (i.e. outside this repository tree).
"""

import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_import_orchestrator_state_machine():
    import orchestrator.state_machine  # noqa: F401


def test_import_orchestrator_agent():
    import orchestrator.agent  # noqa: F401


def test_import_orchestrator_api():
    import orchestrator.api  # noqa: F401


def test_pypi_langgraph_resolves_outside_repo():
    repo_root = os.path.realpath(REPO_ROOT)

    langgraph = importlib.import_module("langgraph")
    # langgraph 1.x is a namespace package: __file__ is None, use __path__.
    locations = [langgraph.__file__] if langgraph.__file__ else list(langgraph.__path__)
    assert locations, "cannot resolve filesystem location of langgraph"
    for loc in locations:
        resolved = os.path.realpath(loc)
        assert not (resolved == repo_root or resolved.startswith(repo_root + os.sep)), (
            f"PyPI langgraph is still shadowed by a repo-local module: {resolved}"
        )

    # The exact import that died pre-fix (BUG-001): local package had no
    # 'graph' submodule, so this raised ModuleNotFoundError.
    graph = importlib.import_module("langgraph.graph")
    graph_resolved = os.path.realpath(graph.__file__)
    assert not graph_resolved.startswith(repo_root + os.sep), (
        f"langgraph.graph resolved inside repo: {graph_resolved}"
    )
