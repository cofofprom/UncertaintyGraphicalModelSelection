import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.utils import generateDiagonalShift


@pytest.mark.parametrize(
    "seed,dim,density",
    [
        (7, 50, 0.3),
        (11, 60, 0.2),
    ],
)
def test_symmetric_unit_diagonal_and_spd(seed, dim, density):
    np.random.seed(seed)
    prec = generateDiagonalShift(dim=dim, density=density)

    assert np.allclose(prec, prec.T, atol=1e-10)
    assert np.allclose(np.diag(prec), np.ones(dim), atol=1e-10)
    assert np.min(np.linalg.eigvalsh(prec)) > 1e-10


def test_off_diagonal_density_matches_request():
    dim = 80
    density = 0.25
    trials = 12
    observed = []

    for seed in range(trials):
        np.random.seed(seed)
        prec = generateDiagonalShift(dim=dim, density=density)
        off_diag = prec[~np.eye(dim, dtype=bool)]
        observed.append(np.count_nonzero(off_diag) / off_diag.size)

    assert np.mean(observed) == pytest.approx(density, abs=0.03)


def test_zero_density_is_still_spd():
    np.random.seed(3)
    prec = generateDiagonalShift(dim=30, density=0.0)

    assert np.allclose(prec, np.eye(30), atol=1e-12)
    assert np.min(np.linalg.eigvalsh(prec)) > 0.0
