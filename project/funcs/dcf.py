from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit

from project.funcs.base import atleast_1d, compute_confusion_matrices, confusion_matrix


@njit(cache=True)
def optimal_bayes_threshold(pi: float, C_fn: float, C_fp: float) -> float:
    return -np.log((pi * C_fn) / ((1 - pi) * C_fp))


def effective_prior(pi_T: float, C_fn: float, C_fp: float) -> float:
    return (pi_T * C_fn) / (pi_T * C_fn + (1 - pi_T) * C_fp)


@njit(cache=True, parallel=True)
def dcf(
    llr: npt.NDArray,
    y_val: npt.NDArray,
    pi: npt.ArrayLike,
    strategy: Literal["optimal"] | Literal["min"] | Literal["manual"],
    Cf_n: float = 1,
    Cf_p: float = 1,
    normalize=True,
    threshold=0.0,
) -> npt.NDArray[np.float64]:
    """
    Compute the Detection Cost Function (DCF) for the given data and priors. This
    function is used to speedup the computation of the DCF values for a range of
    effective priors.

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
        Cf_n (float): The cost of false negative. Defaults to 1.
        Cf_p (float): The cost of false positive. Defaults to 1.
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

        P_fn = cms[:, 1, 0] / cms[:, 1].sum(axis=1)
        P_fp = cms[:, 0, 1] / cms[:, 0].sum(axis=1)

        F = P_fn * Cf_n
        G = P_fp * Cf_p

        for i, p in enumerate(pis):
            denominator = min(p * Cf_n, (1 - p) * Cf_p) if normalize else 1

            res[i] = np.min(p * F + (1 - p) * G) / denominator
    else:
        for i, p in enumerate(pis):
            if strategy == "optimal":
                threshold = optimal_bayes_threshold(p, Cf_n, Cf_p)

            y_pred = llr > threshold

            cm = confusion_matrix(y_val, y_pred)

            P_fn = cm[1, 0] / cm[1].sum()
            P_fp = cm[0, 1] / cm[0].sum()

            res[i] = (p * P_fn * Cf_n + (1 - p) * P_fp * Cf_p) / (
                # Normalize the DCF value by dividing it by the best of the two
                # dummy systems: the one that always accepts a test segment and
                # the one that always rejects it.
                min(p * Cf_n, (1 - p) * Cf_p)
                if normalize
                else 1  # If normalization is not required, return the raw DCF value
            )

    return res


@njit(cache=True, parallel=True)
def bayes_error_plot(
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
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: The effective priors,
            the actual DCF values, and the minimum DCF values.
    """

    effective_prior_log_odds = np.linspace(left, right, num_points)
    effective_priors = 1.0 / (1.0 + np.exp(-effective_prior_log_odds))

    act_dcf = dcf(scores, labels, effective_priors, "optimal")
    min_dcf = dcf(scores, labels, effective_priors, "min")

    return effective_prior_log_odds, act_dcf, min_dcf
