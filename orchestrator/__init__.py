# Orchestrator package (local workflow layer built on PyPI langgraph)
"""Single version source for the whole service (BUG-046).

The installed distribution metadata is authoritative; the literal fallback
only kicks in when running from a source tree with no installed metadata
(e.g. plain ``python -m`` checkouts) and must match pyproject.toml.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("multi-agent-supervisor")
except PackageNotFoundError:  # not installed — running from a bare source tree
    __version__ = "0.2.0"

__all__ = ["__version__"]
