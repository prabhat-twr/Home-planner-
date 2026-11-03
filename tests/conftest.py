import numpy as np
import pytest

from soulhome.synthetic import l_shaped_room


@pytest.fixture(scope="session")
def room():
    return l_shaped_room()


def poly_close(a, b, tol):
    """Same vertex set (any start / order) within tolerance."""
    a, b = np.asarray(a), np.asarray(b)
    assert len(a) == len(b), (a, b)
    for v in a:
        assert np.min(np.linalg.norm(b - v, axis=1)) < tol, (v, b)
