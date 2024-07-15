from functools import partial

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from numba import njit

from project.funcs.base import atleast_1d, quadratic_feature_expansion, vrow


class LogisticRegression:

    def __init__(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_val: npt.NDArray,
        y_val: npt.NDArray,
        quadratic: bool = False,
    ) -> None:
        """
        Initializes the logistic regression classifier.

        Args:
            X_train (npt.NDArray): the training data
            y_train (npt.NDArray): the training labels
            X_val (npt.NDArray): the validation data
            y_val (npt.NDArray): the validation labels
            quadratic (bool, optional): if True, maps the features to a quadratic
                space before training the classifier, defaults to False
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self._quadratic = quadratic

        if quadratic:
            self.X_train = quadratic_feature_expansion(self.X_train)
            self.X_val = quadratic_feature_expansion(self.X_val)

    @property
    def scores(self) -> npt.NDArray:
        """
        Scores of the classifier.
        """
        return self._weights @ self.X_val + self._bias

    @property
    def llr(self) -> npt.NDArray:
        """
        Posterior Log likelihood ratio of the classifier.
        """
        pi = self._prior or np.mean(self.y_train)

        return self.scores - np.log(pi / (1 - pi))

    @property
    def error_rate(self) -> float:
        """
        Error rate measure of the classifier.
        """
        LP = self.scores > 0

        return np.mean(LP != self.y_val)

    def train(self, l: float, prior: float | None = None) -> float:
        """
        Train the logistic regression classifier using the training data and the
        specified hyperparameters.

        Args:
            l (float): the regularization hyperparameter
            prior (float, optional): the prior probability of the positive class,
                if None, the standard logistic regression objective is used,
                otherwise the prior-weighted logistic regression objective is used

        Returns:
            float: the value of the objective function at the optimal point
        """

        self._prior = prior

        log_reg = partial(
            self.objective, DTR=self.X_train, LTR=self.y_train, l=l, prior=prior
        )

        x, f, _ = opt.fmin_l_bfgs_b(
            log_reg,
            np.zeros(self.X_train.shape[0] + 1),
        )

        self._weights, self._bias = x[:-1], x[-1]

        return f

    @staticmethod
    @njit(cache=True)
    def objective(
        v: npt.NDArray[np.float64],
        *,
        prior: float | None,
        DTR: npt.NDArray[np.float64],
        LTR: npt.NDArray[np.int64],
        l: float,
    ) -> tuple[float, npt.NDArray[np.float64] | None]:
        """
        Logistic Regression Objective Function

        Args:
            v (npt.NDArray[np.float64]): the vector of parameters
            prior (float | None): if not None, the prior probability of the positive
                class, computes the prior-weighted logistic regression objective
            DTR (npt.NDArray[np.float64]): the training data
            LTR (npt.NDArray[np.int64]): the training labels
            l (float): the regularization hyperparameter

        Raises:
            ValueError: if the vector of parameters has the wrong shape

        Returns:
            tuple[float, npt.NDArray[np.float64]] | float: the value of the
                objective function and its gradient if approx_grad is False,
                otherwise only the value of the objective function
        """

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

            Xi = np.where(LTR == 1, prior / n_T, (1 - prior) / n_F)
        else:  # Uniform weights, no slow down after jit compilation
            Xi = np.ones_like(LTR) / len(LTR)

        # Logistic regression objective function,
        # J(w, b) = λ/2 * ||w||² + (1/n) ∑_{i=1}^{n} ξᵢ log(1 + exp(-zᵢ(wᵀxᵢ + b)),
        # where ξᵢ = πᴛ / nᴛ if zᵢ = 1, otherwise ξᵢ = (1 - πᴛ) / nꜰ if zᵢ = -1 when
        # prior-weighted logistic regression objective is used, otherwise
        # J(w, b) = λ/2 * ||w||² + (1/n) ∑_{i=1}^{n} log(1 + exp(-zᵢ(wᵀxᵢ + b)),
        # where zᵢ = 1 if cᵢ = 1, otherwise zᵢ = -1 if cᵢ = 0 (i.e. zᵢ = 2cᵢ - 1)
        f = l / 2 * np.linalg.norm(w) ** 2 + np.sum(Xi * np.logaddexp(0, -ZTR * S))

        # Gradient with respect to w, ∇wJ = λw + ∑_{i=1}^{n} ξᵢGᵢxᵢ if
        # prior-weighted logistic regression objective is used,
        # otherwise ∇wJ = λw + (1/n)∑_{i=1}^{n} Gᵢxᵢ
        GW = l * w + (Xi * vrow(G) * DTR).sum(axis=1)

        # Gradient with respect to b, ∇bJ = ∑_{i=1}^{n} ξᵢGᵢ if
        # prior-weighted logistic regression objective is used,
        # otherwise ∇bJ = (1/n)∑_{i=1}^{n} Gᵢ
        Gb = atleast_1d(np.sum(Xi * G))

        return f, np.hstack((GW, Gb))

    def to_json(self) -> dict:
        """
        Serialize the logistic regression classifier to a JSON-serializable dictionary.

        Returns:
            dict: the serialized logistic regression classifier
        """
        return {
            "bias": self._bias,
            "prior": self._prior,
            "quadratic": self._quadratic,
            "weights": self._weights.tolist(),
        }

    @staticmethod
    def from_json(data: dict) -> "LogisticRegression":
        """
        Deserialize a logistic regression classifier from a JSON-serializable dictionary.

        Args:
            data (dict): the serialized logistic regression classifier

        Returns:
            LogisticRegression: the deserialized logistic regression classifier
        """
        log_reg = LogisticRegression.__new__(LogisticRegression)
        log_reg._weights = np.array(data["weights"])
        log_reg._bias = data["bias"]
        log_reg._prior = data["prior"]
        log_reg._quadratic = data["quadratic"]
        return log_reg
