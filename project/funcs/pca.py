import numpy as np
import numpy.typing as npt
from numba import njit

from project.funcs.base import cov, vrow


@njit
def __flip(arr: npt.NDArray) -> npt.NDArray:
    """
    Multiplies the columns of the input array by the sign of the maximum absolute
    value in each column. This ensures that the output of the SVD is deterministic.

    Args:
        arr (npt.NDArray[M, M]): [M x M] input array

    Returns:
        npt.NDArray: [M x M] array with the columns multiplied by the sign of the
        maximum absolute value in each column
    """

    # Get indices of the maximum absolute value in each column
    max_abs_cols = np.argmax(np.abs(arr), axis=0)

    # Compute the indices of the maximum absolute value in each column in the
    shift = np.arange(arr.shape[0])
    idxs = max_abs_cols + shift * arr.shape[1]

    # Get the signs of the maximum absolute value in each column
    signs = np.sign(np.take(arr.T.flatten(), idxs))

    # Multiply the columns by the signs
    return arr * vrow(signs)


@njit
def pca(X: npt.NDArray, m: int | None = None) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Performs Principal Component Analysis on the data matrix X, keeping the
    first m principal components. If m is not provided, all principal components
    are kept. The data is projected onto the PCA directions.

    Args:
        X (NDArray): [N x M] data matrix
        m (int, optional): number of features to keep with m <= M

    Returns:
        The eigenvectors and PCA data matrix

        directions: [M x m] matrix with the eigenvectors of the covariance matrix
        PCA_data: [N x m] matrix with the data projected onto the PCA directions
    """

    # Since the covariance matrix is semi-definite positive the sorted
    # (descending) eigenvectors can be obtained by singular value decomposition
    # of the covariance matrix, C = U S V^T where V^T == U^T
    U, _, _ = np.linalg.svd(cov(X.T))

    # Flip the singular vectors to ensure deterministic output
    U = __flip(U)

    # Take the first m eigenvectors
    directions = U[:, :m]

    # Make arrays contiguous for better performance
    directions = np.ascontiguousarray(directions)
    X = np.ascontiguousarray(X)

    return directions, np.dot(X, directions)
