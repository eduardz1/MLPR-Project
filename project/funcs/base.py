from typing import Generator

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def vcol(vec: npt.NDArray) -> npt.NDArray:
    return vec.reshape(-1, 1)


@njit(cache=True)
def vrow(vec: npt.NDArray) -> npt.NDArray:
    return vec.reshape(1, -1)


def load_data(path: str) -> tuple[npt.NDArray, npt.NDArray]:
    """Load the data from the given path

    Args:
        path (str): Path to the data file

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Features and labels
    """
    dataset = np.loadtxt(path, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    return X, y


@njit(cache=True)
def mean(x, axis=None):
    if axis is None:
        return np.sum(x, axis) / np.prod(x.shape)
    else:
        return np.sum(x, axis) / x.shape[axis]


@njit(cache=True)
def cov(X: npt.NDArray) -> npt.NDArray:
    """Compute the covariance matrix of the given data. Centers the data first.

    Args:
        X (npt.NDArray): The data

    Returns:
        npt.NDArray: The covariance matrix
    """
    DC = X - vcol(mean(X, 1))

    return (DC @ DC.T) / float(X.shape[1])


def corr(X: npt.NDArray) -> npt.NDArray:
    """Compute the correlation matrix of the given data

    Args:
        X (npt.NDArray): The data

    Returns:
        npt.NDArray: The correlation matrix
    """
    return (C := cov(X)) / (vcol(C.diagonal() ** 0.5) * vrow(C.diagonal() ** 0.5))


@njit(cache=True)
def atleast_1d(x):
    """
    Workaround for the numba issue with `np.atleast_1d`
    https://github.com/numba/numba/issues/4202
    """
    zero = np.zeros((1,), dtype=np.bool_)
    return x + zero


@njit(cache=True)
def atleast_2d(x):
    """
    Workaround for the numba issue with `np.atleast_2d`
    https://github.com/numba/numba/issues/4202
    """
    zero = np.zeros((1, 1), dtype=np.bool_)
    return x + zero


@njit(cache=True)
def split_db_2to1(
    D: npt.NDArray, L: npt.NDArray, seed=0
) -> tuple[tuple[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]:
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


@njit(cache=True)
def confusion_matrix(y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray[np.int32]:
    """Compute the confusion matrix for the given true and predicted labels

    Args:
        y_true (npt.NDArray): The true labels
        y_pred (npt.NDArray): The predicted labels

    Returns:
        npt.NDArray: The confusion matrix with the following structure:
            [[TN, FP],
             [FN, TP]]
    """
    return np.array(
        [
            [
                np.sum(np.logical_and(y_true == 0, y_pred == 0)),
                np.sum(np.logical_and(y_true == 0, y_pred == 1)),
            ],
            [
                np.sum(np.logical_and(y_true == 1, y_pred == 0)),
                np.sum(np.logical_and(y_true == 1, y_pred == 1)),
            ],
        ]
    )


@njit
def yield_confusion_matrices(
    y_true: npt.NDArray, thresholds: npt.NDArray
) -> Generator[npt.NDArray[np.int32], None, None]:
    indices = np.argsort(thresholds)
    ts = thresholds[indices]
    sorted_y_val = y_true[indices]

    y_pred = np.ones_like(y_true)

    TN = 0
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    FP = len(y_true) - TP
    FN = 0

    for i in range(1, len(ts)):
        y_pred[i - 1] = 0

        if sorted_y_val[i - 1] == 1:
            TP -= 1
            FN += 1
        else:
            FP -= 1
            TN += 1

        yield np.array([[TN, FP], [FN, TP]])


def quadratic_feature_expansion(X):
    X = X.T

    # Compute the outer product for each vector with itself using einsum
    outer_products = np.einsum("ij,ik->ijk", X, X)

    flattened_outer_products = outer_products.reshape(X.shape[0], -1)
    X_panded = np.concatenate([flattened_outer_products, X], axis=1)

    return X_panded.T
