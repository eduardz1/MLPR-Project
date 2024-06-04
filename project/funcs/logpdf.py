import numpy as np
import numpy.typing as npt


def _check_params(X: npt.NDArray, mu: npt.NDArray, C: npt.NDArray) -> None:
    # fmt: off
    assert len(X.shape) == 2, f"X must be a 2D array, got {len(X.shape)}"
    assert len(mu) == X.shape[0], f"mu must have the same number of rows as X ({X.shape[0]}), got {len(mu)}"
    assert len(C.shape) == 2, f"C must be a 2D array, got {len(C.shape)}"
    assert C.shape[0] == C.shape[1], f"C must be a square matrix, got {C.shape[0]}x{C.shape[1]}"
    assert C.shape[0] == X.shape[0], f"C must have the same number of rows as X ({X.shape[0]}), got {C.shape[0]}x{C.shape[1]}"
    # fmt: on


def log_pdf(X: npt.NDArray, mu: npt.NDArray, C: npt.NDArray) -> npt.NDArray:
    """Calculates the logarithm of the multivariate gaussian density function

    Args:
        X (NDArray): [M x N] data matrix
        mu (ArrayLike): [M x 1] mean vector
        C (NDArray): [M x M] covariance matrix

    Returns:
        NDArray: [N x 1] logarithm of the multivariate gaussian density function
    """
    X = np.atleast_2d(X)
    mu = np.atleast_1d(mu)
    C = np.atleast_2d(C)

    _check_params(X, mu, C)

    return -0.5 * (
        X.shape[0] * np.log(2 * np.pi)
        + np.linalg.slogdet(C)[1]
        + np.einsum("ij,ji->i", np.dot((X - mu).T, np.linalg.inv(C)), (X - mu))
    )


def log_likelihood(X: npt.NDArray, mu: npt.NDArray, C: npt.NDArray) -> float:
    """Calculates the log likelihood of the data given the parameters

    Args:
        X (NDArray): [M x N] data matrix
        mu (ArrayLike): [M x 1] mean vector
        C (NDArray): [M x M] covariance matrix

    Returns:
        float: log likelihood of the data given the parameters
    """
    return np.sum(log_pdf(X, mu, C))
