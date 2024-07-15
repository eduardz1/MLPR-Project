import numpy as np
import numpy.typing as npt
import scipy.special as sp
from numba import njit

from project.funcs.base import atleast_2d


@njit(cache=True)
def _check_params(X: npt.NDArray, mu: npt.NDArray, C: npt.NDArray) -> None:
    # fmt: off
    assert len(X.shape) == 2, f"X must be a 2D array, got {len(X.shape)}"
    assert len(mu) == X.shape[0], f"mu must have the same number of rows as X ({X.shape[0]}), got {len(mu)}"
    assert len(C.shape) == 2, f"C must be a 2D array, got {len(C.shape)}"
    assert C.shape[0] == C.shape[1], f"C must be a square matrix, got {C.shape[0]}x{C.shape[1]}"
    assert C.shape[0] == X.shape[0], f"C must have the same number of rows as X ({X.shape[0]}), got {C.shape[0]}x{C.shape[1]}"
    # fmt: on


@njit(cache=True)
def log_pdf_gaussian(X: npt.NDArray, mu: npt.NDArray, C: npt.NDArray) -> npt.NDArray:
    """Calculates the logarithm of the multivariate gaussian density function

    Args:
        X (NDArray): [M x N] data matrix
        mu (NDArray): [M x 1] mean vector
        C (NDArray): [M x M] covariance matrix

    Returns:
        NDArray: [N x 1] logarithm of the multivariate gaussian density function
    """
    X = atleast_2d(X)
    mu = atleast_2d(mu)
    C = atleast_2d(C)

    _check_params(X, mu, C)

    return -0.5 * (
        X.shape[0] * np.log(2 * np.pi)
        + np.linalg.slogdet(C)[1]
        + ((X - mu) * (np.linalg.inv(C) @ (X - mu))).sum(0)
    )


def log_pdf_gmm(
    X: npt.NDArray, gmm: list[tuple[npt.NDArray, npt.NDArray, npt.NDArray]]
) -> npt.NDArray:
    """Calculates the logarithm of the gaussian mixture model density function

    Args:
        X (NDArray): [M x N] data matrix
        gmm (List[Tuple[NDArray, NDArray, NDArray]]): list of tuples containing
            the weights, means and covariances of the GMM

    Returns:
        NDArray: [N x K] logarithm of the gaussian mixture model density function
    """
    S = [log_pdf_gaussian(X, mu, C) + np.log(w) for w, mu, C in gmm]

    S = np.vstack(S)

    return sp.logsumexp(S, axis=0)  # type: ignore
