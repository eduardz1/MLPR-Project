from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit

from project.funcs.base import atleast_1d, compute_confusion_matrices, confusion_matrix


@njit(cache=True)
def optimal_bayes_threshold(pi: float, cost_fn: float, cost_fp: float) -> float:
    """
    Compute the optimal threshold for the given prior and cost of false negative
    and false positive.

    Args:
        pi (float): The prior probability of a genuine sample.
        cost_fn (float): The cost of false negative.
        cost_fp (float): The cost of false positive.

    Returns:
        float: The optimal threshold.
    """
    return -np.log((pi * cost_fn) / ((1 - pi) * cost_fp))


def effective_prior(pi: float, cost_fn: float, cost_fp: float) -> float:
    """
    Compute the effective prior for the given prior and cost of false negative
    and false positive.

    Args:
        pi (float): The prior probability of a genuine sample.
        cost_fn (float): The cost of false negative.
        cost_fp (float): The cost of false positive.

    Returns:
        float: The effective prior.
    """
    return (pi * cost_fn) / (pi * cost_fn + (1 - pi) * cost_fp)


@njit(cache=True, parallel=True)
def dcf(
    llr: npt.NDArray,
    y_val: npt.NDArray,
    pi: npt.ArrayLike,
    strategy: Literal["optimal"] | Literal["min"] | Literal["manual"],
    cost_fn: float = 1,
    cost_fp: float = 1,
    normalize=True,
    threshold=0.0,
) -> npt.NDArray[np.float64]:
    """
    Compute the binary Detection Cost Function (DCF) (Bayes Risk) for the given
    data, priors and cost of false negative and false positive. Can compute the
    actual DCF value by using the optimal threshold, the minimum DCF value, or a
    DCF value for a given threshold.

    Args:
        llr (NDArray): The log-likelihood ratio values.
        y_val (NDArray): The true labels.
        pi (npt.ArrayLike): The prior probability of a genuine sample or a list
            of priors to use.
        strategy (
            Literal["optimal"]
            | Literal["min"]
            | Literal["manual"],
        ): The threshold strategy to use, either "optimal", "min", or "manual".
            Use "optimal" to compute the optimal threshold, "min" to compute the
            minimum DCF value, and "manual" to use the given threshold.
        cost_fn (float): The cost of false negative. Defaults to 1.
        cost_fp (float): The cost of false positive. Defaults to 1.
        normalize (bool, optional): Whether to normalize the DCF value.
            Defaults to True.
        threshold (float, optional): The threshold to use if strategy is "manual".
            Does not have any effect if strategy is not "manual". Defaults to 0.0.

    Returns:
        NDArray: The DCF values for the given priors.
    """

    pis = atleast_1d(pi)
    res = np.empty(len(pis))

    if strategy == "min":
        cms = compute_confusion_matrices(y_val, llr)

        p_fn = cms[:, 1, 0] / cms[:, 1].sum(axis=1)
        p_fp = cms[:, 0, 1] / cms[:, 0].sum(axis=1)

        F = p_fn * cost_fn
        P = p_fp * cost_fp

        for i, _pi in enumerate(pis):
            denominator = min(_pi * cost_fn, (1 - _pi) * cost_fp) if normalize else 1

            res[i] = np.min(_pi * F + (1 - _pi) * P) / denominator
    else:
        for i, _pi in enumerate(pis):
            if strategy == "optimal":
                threshold = optimal_bayes_threshold(_pi, cost_fn, cost_fp)

            y_pred = llr > threshold

            cm = confusion_matrix(y_val, y_pred)

            p_fn = cm[1, 0] / (cm[1, 0] + cm[1, 1])
            p_fp = cm[0, 1] / (cm[0, 0] + cm[0, 1])

            res[i] = (_pi * p_fn * cost_fn + (1 - _pi) * p_fp * cost_fp) / (
                # Normalize the DCF value by dividing it by the best of the two
                # dummy systems: the one that always accepts a test segment and
                # the one that always rejects it.
                min(_pi * cost_fn, (1 - _pi) * cost_fp)
                if normalize
                else 1  # If normalization is not required, return the raw DCF value
            )

    return res


@njit(cache=True)
def bayes_error(
    scores: npt.NDArray,
    labels: npt.NDArray,
    left: float = -4,
    right: float = 4,
    num_points: int = 100,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Utility function to compute the actual and minimum DCF values for a range
    of effective priors.

    Args:
        scores (npt.NDArray): The log-likelihood ratio values.
        labels (npt.NDArray): The true labels.
        left (float, optional): The left bound of the effective prior range.
            Defaults to -3.
        right (float, optional): The right bound of the effective prior range.
            Defaults to 3.
        num_points (int, optional): The number of points to use in the range.
            Defaults to 21.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]: The effective priors,
            the actual DCF values, and the minimum DCF values.
    """

    effective_prior_log_odds = np.linspace(left, right, num_points)
    effective_priors = 1.0 / (1.0 + np.exp(-effective_prior_log_odds))

    act_dcf = dcf(scores, labels, effective_priors, "optimal")
    min_dcf = dcf(scores, labels, effective_priors, "min")

    return effective_prior_log_odds, act_dcf, min_dcf
