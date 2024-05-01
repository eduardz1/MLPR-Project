import numpy as np
import numpy.typing as npt


def pca(X: npt.NDArray, m: int | None = None) -> tuple[npt.NDArray, npt.NDArray]:
    """Performs Principal Component Analysis on the data matrix X

    Args:
        X (NDArray): [N x M] data matrix
        m (int, optional): number of features to keep with m <= M

    Returns:
        The eigenvectors and PCA data matrix

        PCA_eigvec: [M x m] matrix with the eigenvectors of the covariance matrix
        PCA_data: [N x m] matrix with the PCA data
    """

    _, eigvec = np.linalg.eigh(np.cov(X.T, bias=True))

    # Reverse so that the eigen vectors are sorted in
    # decreasing order and take the first m eigenvectors
    PCA_eigvec = eigvec[:, ::-1][:, :m]

    return PCA_eigvec, np.dot(X, PCA_eigvec)
