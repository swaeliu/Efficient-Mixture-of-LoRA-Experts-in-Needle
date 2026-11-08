#!/usr/bin/env python
"""Make the local Needle implementation importable from the active environment.

Needle (the 10-714 course framework) is not distributed as a package, so this
script writes a ``needle_local.pth`` file into the active interpreter's
site-packages that points at the directory containing the ``needle/`` package.
Nothing in the coursework directory is modified or copied.

Usage (with the project virtualenv activated)::

    python scripts/link_needle.py                 # auto-detect ../hw2/python
    python scripts/link_needle.py /path/to/python # explicit directory containing needle/
    python scripts/link_needle.py --remove        # undo

The alternative, which needs no .pth file, is to export
``NEEDLE_PYTHON_DIR=/path/to/python`` before running anything; ``molora``
falls back to that variable when ``import needle`` fails.
"""
from __future__ import annotations

import argparse
import os
import sys
import sysconfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CANDIDATES = [
    os.environ.get("NEEDLE_PYTHON_DIR", ""),
    os.path.join(os.path.dirname(REPO_ROOT), "hw2", "python"),
    os.path.expanduser("~/projects/dlsys/hw2/python"),
]
PTH_NAME = "needle_local.pth"


def is_needle_dir(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "needle", "autograd.py"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("needle_python_dir", nargs="?", help="directory that contains the needle/ package")
    ap.add_argument("--remove", action="store_true", help="remove the .pth file instead of writing it")
    args = ap.parse_args()

    site = sysconfig.get_paths()["purelib"]
    pth = os.path.join(site, PTH_NAME)

    if args.remove:
        if os.path.exists(pth):
            os.remove(pth)
            print(f"removed {pth}")
        else:
            print(f"nothing to remove at {pth}")
        return 0

    candidates = [args.needle_python_dir] if args.needle_python_dir else DEFAULT_CANDIDATES
    target = next((os.path.abspath(c) for c in candidates if is_needle_dir(c)), None)
    if target is None:
        print("error: could not find a directory containing needle/autograd.py; tried:", file=sys.stderr)
        for c in candidates:
            if c:
                print(f"  {c}", file=sys.stderr)
        print("pass the path explicitly: python scripts/link_needle.py /path/to/python", file=sys.stderr)
        return 1

    with open(pth, "w") as f:
        f.write(target + "\n")
    print(f"wrote {pth} -> {target}")

    # Verify in a fresh interpreter so the .pth is actually honoured.
    import subprocess

    code = "import needle, os; print('needle importable from', os.path.dirname(needle.__file__))"
    rc = subprocess.call([sys.executable, "-c", code])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
