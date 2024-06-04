import numpy as np
import numpy.typing as npt
import scipy

from project.funcs.base import cov, vcol, vrow


def lda(
    X: npt.NDArray, y: npt.ArrayLike, m: int | None = None
) -> tuple[npt.NDArray, npt.NDArray]:
    """Performs Linear Discriminant Analysis on the data matrix X

    Args:
        X (NDArray): [N x M] data matrix
        y (ArrayLike): [N x 1] label vector
        m (int, optional): number of features to keep with m <= M

    Returns:
        The directions and LDA data matrix

        directions: [M x m] matrix with the eigenvectors of the covariance matrix
        LDA_data: [N x m] matrix with the data projected onto the LDA directions
    """

    labels = np.unique(y)
    means = np.array([np.mean(X[y == c], axis=0) for c in labels])
    global_mean = np.mean(X, axis=0)
    weights = np.array([len(X[y == c]) for c in labels])

    # fmt: off
    # Compute the between-class covariance matrix
    Sb = np.average(
        [vcol(mu := (means[c] - global_mean)) @ vrow(mu) for c in labels],
        axis=0,
        weights=weights,
    )

    # Compute the within-class covariance matrix
    Sw = np.average(
        [cov(X[y == c].T) for c in labels],
        axis=0,
        weights=weights
    )
    # fmt: on

    # Compute the eigenvectors of the generalized eigenvalue problem
    _, eigvec = scipy.linalg.eigh(Sb, Sw)

    # Reverse so that the eigen vectors are sorted in
    # decreasing order and take the first m eigenvectors
    directions = eigvec[:, ::-1][:, :m]

    return directions, np.dot(X, directions)
