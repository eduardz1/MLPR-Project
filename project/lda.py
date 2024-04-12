import numpy as np
import scipy


def lda(X, y, m):
    """Performs Linear Discriminant Analysis on the data matrix X

    Args:
        X (array_like): [N x M] data matrix
        y (array_like): [N x 1] label vector
        m (uint, optional): number of features to keep with m <= M

    Returns:
        The eigenvectors and LDA data matrix

        LDA_eigvec: [M x m] matrix with the eigenvectors of the covariance matrix
        LDA_data: [N x m] matrix with the LDA data
    """

    unique_labels = np.unique(y)
    means = np.array([np.mean(X[y == c], axis=0) for c in unique_labels])
    global_mean = np.mean(X, axis=0)
    weights = np.array([len(X[y == c]) for c in unique_labels])

    # fmt: off
    # Compute the between-class covariance matrix
    Sb = np.average(
        [
            np.outer(means[c] - global_mean, means[c] - global_mean)
            for c in unique_labels
        ],
        axis=0,
        weights=weights
    )

    # Compute the within-class covariance matrix
    Sw = np.average(
        [   # ndmin=2 to handle the case where there is only one dimension
            np.array(np.cov(X[y == c].T), ndmin=2)
            for c in unique_labels
        ],
        axis=0,
        weights=weights
    )
    # fmt: on

    # Compute the eigenvectors of the generalized eigenvalue problem
    _, eigvec = scipy.linalg.eigh(Sb, Sw)

    # Reverse so that the eigen vectors are sorted in
    # decreasing order and take the first m eigenvectors
    LDA_eigvec = eigvec[:, ::-1][:, :m]

    return LDA_eigvec, np.dot(X, LDA_eigvec)
