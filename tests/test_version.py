"""Pin __version__ in __init__.py and pyproject.toml together.

server-k6sq: 0.4.3 wheel shipped with __version__='0.4.2' because the
release procedure bumped pyproject.toml but not src/veep/__init__.py.
This test fails the build if those two strings drift again.
"""
from __future__ import annotations

import importlib.metadata

import veep


def test_version_matches_distribution_metadata():
    """veep.__version__ must equal the installed distribution's version.

    If this fails, one of the two version sources of truth was bumped
    without the other. Fix both:
      - pyproject.toml: version = "X.Y.Z"
      - src/veep/__init__.py: __version__ = "X.Y.Z"
    """
    assert veep.__version__ == importlib.metadata.version("veep"), (
        f"veep.__version__ ({veep.__version__}) != "
        f"importlib.metadata.version('veep') "
        f"({importlib.metadata.version('veep')}). "
        f"pyproject.toml and src/veep/__init__.py disagree. "
        f"Bump both in lockstep at release time."
    )
