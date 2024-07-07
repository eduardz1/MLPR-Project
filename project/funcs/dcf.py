from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit

from project.funcs.base import confusion_matrix, yield_confusion_matrices


@njit(cache=True)
def optimal_bayes_threshold(pi: float, C_fn: float, C_fp: float) -> float:
    return -np.log((pi * C_fn) / ((1 - pi) * C_fp))


def effective_prior(pi_T: float, C_fn: float, C_fp: float) -> float:
    return (pi_T * C_fn) / (pi_T * C_fn + (1 - pi_T) * C_fp)


@njit(cache=True, parallel=True)
def dcf(
    llr: npt.NDArray,
    y_val: npt.NDArray,
    pi: float,
    Cf_n: float,
    Cf_p: float,
    strategy: Literal["optimal"] | Literal["min"] | Literal["manual"],
    normalize=True,
    threshold=0.0,
) -> float:
    """
    Compute the Detection Cost Function (DCF) for the given data and priors.

    Args:
        llr (NDArray): The log-likelihood ratio values.
        y_val (NDArray): The true labels.
        pi (float): The prior probability of a genuine sample.
        Cf_n (float): The cost of false negative.
        Cf_p (float): The cost of false positive.
        strategy (
            Literal["optimal"]
            | Literal["min"]
            | Literal["manual"],
        ): The threshold strategy to use, either "optimal", "min", or "manual".
            Use "optimal" to compute the optimal threshold, "min" to compute the
            minimum DCF value, and "manual" to use the given threshold.
        normalize (bool, optional): Whether to normalize the DCF value.
            Defaults to True.
        threshold (float, optional): The threshold to use if strategy is "manual".
            Does not have any effect if strategy is not "manual". Defaults to 0.0.

    Returns:
        float: The DCF value.
    """

    if strategy == "min":
        # Returns the minimum DCF value calculated over all the possible
        # threhsolds (taken from the log likelihood ratios)
        return min(
            [
                (
                    pi * (cm[1, 0] / cm[1].sum()) * Cf_n
                    + (1 - pi) * (cm[0, 1] / cm[0].sum()) * Cf_p
                )
                / (min(pi * Cf_n, (1 - pi) * Cf_p) if normalize else 1)
                for cm in yield_confusion_matrices(y_val, llr)
            ]
        )
    else:
        threshold = (
            optimal_bayes_threshold(pi, Cf_n, Cf_p)
            if strategy == "optimal"
            else threshold  # if strategy == "manual"
        )

        y_pred = llr > threshold

        cm = confusion_matrix(y_val, y_pred)

        P_fn = cm[1, 0] / cm[1].sum()
        P_fp = cm[0, 1] / cm[0].sum()

        return (pi * P_fn * Cf_n + (1 - pi) * P_fp * Cf_p) / (
            # Normalize the DCF value by dividing it by the best of the two
            # dummy systems: the one that always accepts a test segment and
            # the one that always rejects it.
            min(pi * Cf_n, (1 - pi) * Cf_p)
            if normalize
            else 1  # If normalization is not required, return the raw DCF value
        )


@njit(cache=True, parallel=True)
def bayes_error_plot(
    scores: npt.NDArray,
    labels: npt.NDArray,
    left: float = -4,
    right: float = 4,
    num_points: int = 100,
) -> tuple[npt.ArrayLike, list, list]:
    """
    Utility function to compute the actual and minimum DCF values for a range
    of effective priors.

    Args:
        scores (npt.NDArray): The log-likelihood ratio values.
        labels (npt.NDArray): The true labels.
        left (float, optional): The left bound of the effective prior range.
            Defaults to -4.
        right (float, optional): The right bound of the effective prior range.
            Defaults to 4.
        num_points (int, optional): The number of points to use in the range.
            Defaults to 100.

    Returns:
        Tuple[npt.ArrayLike, list, list]: The effective priors,
            the actual DCF values, and the minimum DCF values.
    """

    effective_prior_log_odds = np.linspace(left, right, num_points)
    effective_priors = 1.0 / (1.0 + np.exp(-effective_prior_log_odds))

    act_dcf = []
    min_dcf = []

    for effective_prior in effective_priors:
        act_dcf.append(dcf(scores, labels, effective_prior, 1, 1, "optimal"))
        min_dcf.append(dcf(scores, labels, effective_prior, 1, 1, "min"))

    return effective_prior_log_odds, act_dcf, min_dcf
