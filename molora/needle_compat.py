"""Locate and import the Needle framework.

Needle is the user's completed 10-714 coursework implementation and is not a
PyPI package. Resolution order:

1. ``import needle`` succeeds as-is (``scripts/link_needle.py`` wrote a
   ``.pth`` file, or needle is otherwise on ``sys.path``).
2. ``NEEDLE_PYTHON_DIR`` environment variable: the directory *containing* the
   ``needle/`` package.
3. The conventional sibling layout ``<repo>/../hw2/python``.
"""
from __future__ import annotations

import os
import sys
from types import ModuleType

__all__ = ["import_needle", "needle_info", "NEEDLE_ENV_VAR"]

NEEDLE_ENV_VAR = "NEEDLE_PYTHON_DIR"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _candidate_dirs() -> list[str]:
    cands = []
    env = os.environ.get(NEEDLE_ENV_VAR)
    if env:
        cands.append(env)
    cands.append(os.path.join(os.path.dirname(_REPO_ROOT), "hw2", "python"))
    return cands


def _is_needle_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "needle", "autograd.py"))


def import_needle() -> ModuleType:
    """Import and return the ``needle`` module, adding it to ``sys.path`` if needed."""
    try:
        import needle  # noqa: F401

        return needle
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        if exc.name != "needle":
            raise

    for cand in _candidate_dirs():
        if _is_needle_dir(cand):
            sys.path.insert(0, os.path.abspath(cand))
            import needle  # noqa: F401

            return needle

    raise ImportError(
        "Could not import the Needle framework. Either run "
        "`python scripts/link_needle.py /path/to/python` (directory containing needle/) "
        f"or set {NEEDLE_ENV_VAR} to that directory. See docs/needle_environment.md."
    )


def needle_info() -> dict:
    """Return a small dict describing the Needle implementation in use."""
    import needle

    path = os.path.dirname(os.path.abspath(needle.__file__))
    backend = getattr(needle.autograd, "array_api", None)
    return {
        "needle_path": path,
        "array_api": getattr(backend, "__name__", str(backend)),
        "devices": [repr(d) for d in needle.all_devices()],
    }
