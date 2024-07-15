from typing import Callable

import numpy as np
import numpy.typing as npt

from project.classifiers.logistic_regression import LogisticRegression


def extract_fold(X: npt.NDArray, i: int, K: int):
    """
    Extract the i-th fold from the given data

    Args:
        X (npt.NDArray): The data
        i (int): The fold to extract
        K (int): The number of folds

    Returns:
        tuple[npt.NDArray, npt.NDArray]: The training and validation data
    """
    return np.ascontiguousarray(
        np.hstack([X[j::K] for j in range(K) if j != i])
    ), np.ascontiguousarray(X[i::K])


def kfolds(
    scores: list[npt.NDArray],
    y_val: npt.NDArray,
    progress: Callable,
    pi: float,
    K: int,
):
    calibrated_scores = []
    calibrated_labels = []

    for i in range(K):
        LCAL, LVAL = extract_fold(y_val, i, K)

        SCAL_SVAL = [extract_fold(s, i, K) for s in scores]

        SCAL = np.vstack([scal for scal, _ in SCAL_SVAL])
        SVAL = np.vstack([sval for _, sval in SCAL_SVAL])

        log_reg = LogisticRegression()

        # We don't need to regularize the lambda term
        log_reg.fit(SCAL, LCAL, l=0, prior=pi).scores(SVAL)

        calibrated_scores.append(log_reg.llr)
        calibrated_labels.append(LVAL)

        progress()

    return np.hstack(calibrated_scores), np.hstack(calibrated_labels)
