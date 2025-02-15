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
def atleast_1d(x) -> npt.NDArray:
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
            [sum(~y_true & ~y_pred), sum(~y_true & y_pred)],
            [sum(y_true & ~y_pred), sum(y_true & y_pred)],
        ]
    )


@njit(cache=True, parallel=True)
def compute_confusion_matrices(
    y_true: npt.NDArray, thresholds: npt.NDArray
) -> npt.NDArray[np.int32]:
    """
    Efficient way of generating confusion matrices for a set of thresholds
    without computing the entire confusion matrix for each threshold.

    Args:
        y_true (npt.NDArray): The true labels
        thresholds (npt.NDArray): The thresholds to use

    Returns:
        npt.NDArray: [N, 2, 2] The confusion matrices for each threshold
    """
    indices = np.argsort(thresholds)
    ts = thresholds[indices]
    sorted_y_val = y_true[indices]

    y_pred = np.ones_like(y_true)

    TN = 0
    TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    FP = len(y_true) - TP
    FN = 0

    cms = np.empty((len(ts) - 1, 2, 2), dtype=np.int32)
    for i in range(len(ts) - 1):
        y_pred[i] = 0

        if sorted_y_val[i] == 1:
            TP -= 1
            FN += 1
        else:
            FP -= 1
            TN += 1

        cms[i, 0, 0] = TN
        cms[i, 0, 1] = FP
        cms[i, 1, 0] = FN
        cms[i, 1, 1] = TP

    return cms


def quadratic_feature_expansion(X: npt.NDArray[np.float64]) -> npt.NDArray:
    """
    Applies quadratic feature expansion to the given data, mapping X to

             __          __
             | vec(x x^T) |
    phi(x) = |     x      |
             --          --

    Args:
        X (npt.NDArray[np.float64]): The data matrix

    Returns:
        npt.NDArray: The expanded data matrix
    """
    X = X.T

    X_panded = [np.concatenate([np.outer(x, x).flatten(), x]) for x in X]

    return np.array(X_panded).T
