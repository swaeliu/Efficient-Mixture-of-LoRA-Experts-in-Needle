"""Shared pytest fixtures. Importing molora first guarantees needle is importable."""
import numpy as np
import pytest

import molora  # noqa: F401  (resolves the needle import path)


@pytest.fixture(autouse=True)
def _seed_numpy():
    """Needle's initialisers draw from numpy's global RNG; seed it for every test."""
    np.random.seed(0)
    yield
