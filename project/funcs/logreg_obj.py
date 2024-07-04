import numpy as np
import numpy.typing as npt
from numba import njit

from project.funcs.base import vrow


@njit
def __logreg_obj(
    v: npt.NDArray[np.float64],
    *,
    prior: float | None,
    approx_grad: bool,
    DTR: npt.NDArray[np.float64],
    LTR: npt.NDArray[np.int64],
    l: float,
) -> tuple[float, npt.NDArray[np.float64] | None]:
    if v.shape[0] != (DTR.shape[0] + 1):
        raise ValueError(
            "The vector of parameters has the wrong shape, expected (n,) where"
            "n is the number of features in the training data."
        )

    w, b = v[:-1], v[-1]
    ZTR = 2 * LTR - 1
    S = (w @ np.ascontiguousarray(DTR) + b).ravel()
    G = -ZTR / (1 + np.exp(ZTR * S))

    if prior is not None:
        n_T = np.sum(LTR == 1)
        n_F = len(LTR) - n_T

        weights = np.where(LTR == 1, prior / n_T, (1 - prior) / n_F)
    else:  # Uniform weights, no slow down after jit compilation
        weights = np.ones_like(LTR) / len(LTR)

    # Logistic regression objective function,
    # J(w, b) = λ/2 * ||w||² + (1/n) ∑_{i=1}^{n} ξᵢ log(1 + exp(-zᵢ(wᵀxᵢ + b)),
    # where ξᵢ = πᴛ / nᴛ if zᵢ = 1, otherwise ξᵢ = (1 - πᴛ) / nꜰ if zᵢ = -1 when
    # prior-weighted logistic regression objective is used, otherwise
    # J(w, b) = λ/2 * ||w||² + (1/n) ∑_{i=1}^{n} log(1 + exp(-zᵢ(wᵀxᵢ + b)),
    # where zᵢ = 1 if cᵢ = 1, otherwise zᵢ = -1 if cᵢ = 0 (i.e. zᵢ = 2cᵢ - 1)
    f = l / 2 * np.linalg.norm(w) ** 2 + np.sum(weights * np.logaddexp(0, -ZTR * S))

    if not approx_grad:
        # fmt: off
        vgrad = np.array(
            [
                # Gradient with respect to w, ∇wJ = λw + ∑_{i=1}^{n} ξᵢGᵢxᵢ if
                # prior-weighted logistic regression objective is used,
                # otherwise ∇wJ = λw + (1/n)∑_{i=1}^{n} Gᵢxᵢ
                *(l * w + np.sum(weights * vrow(G) * DTR)),

                # Gradient with respect to b, ∇bJ = ∑_{i=1}^{n} ξᵢGᵢ if
                # prior-weighted logistic regression objective is used,
                # otherwise ∇bJ = (1/n)∑_{i=1}^{n} Gᵢ
                np.sum(weights * G),
            ]
        )
        # fmt: on
        return f, vgrad

    return f, None


def logreg_obj(
    v: npt.NDArray[np.float64],
    *,
    prior: float | None,
    approx_grad: bool,
    DTR: npt.NDArray[np.float64],
    LTR: npt.NDArray[np.int64],
    l: float,
) -> tuple[float, npt.NDArray[np.float64]] | float:
    """
    Logistic Regression Objective Function

    Args:
        v (npt.NDArray[np.float64]): the vector of parameters
        prior (float | None): if not None, the prior probability of the positive
            class, computes the prior-weighted logistic regression objective
        approx_grad (bool): if True, only the result to the objective function
            is returned, if False, the gradient is also returned
        DTR (npt.NDArray[np.float64]): the training data
        LTR (npt.NDArray[np.int64]): the training labels
        l (float): the regularization hyperparameter

    Returns:
        tuple[float, npt.NDArray[np.float64]] | float: the value of the
            objective function and its gradient if approx_grad is False,
            otherwise only the value of the objective function
    """
    # This function wraps the real implementation to manage etheroegeneous
    # return types in the numba.jit compiled function
    result = __logreg_obj(
        v,
        prior=prior,
        approx_grad=approx_grad,
        DTR=DTR,
        LTR=LTR,
        l=l,
    )

    if approx_grad:
        return result[0]

    return result  # type: ignore
