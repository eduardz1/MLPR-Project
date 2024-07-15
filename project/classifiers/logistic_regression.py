import json
from functools import partial
from io import TextIOWrapper

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from numba import njit

from project.classifiers.classifier import Classifier
from project.funcs.base import atleast_1d, quadratic_feature_expansion, vrow


class LogisticRegression(Classifier):
    """
    Logistic Regression Classifier.

    Attributes:
        llr (npt.NDArray[np.float64]): The posterior log likelihood ratio of the classifier.
        accuracy (float): The accuracy of the classifier.
        error_rate (float): The error rate of the classifier.

        _bias (float): The bias term of the classifier.
        _f (float): The value of the objective function at the optimal point.
        _fitted (bool): Whether the classifier has been fitted or not.
        _prior (float): The prior probability of the positive class.
        _quadratic (bool): Whether to map the features to a quadratic space before
            training the classifier.
        _weights (npt.NDArray[np.float64]): The weights of the classifier.
        _S (npt.NDArray[np.float64]): The scores of the classifier.
    """

    def __init__(self, quadratic: bool = False) -> None:
        """
        Initializes the logistic regression classifier.

        Args:
            quadratic (bool, optional): if True, maps the features to a quadratic
                space before training the classifier, defaults to False
        """
        self._quadratic = quadratic

    def scores(self, X):
        if not self._fitted:
            raise ValueError("Classifier has not been fitted yet.")

        if self._quadratic:
            X = quadratic_feature_expansion(X)

        self._S = self._weights.T @ X + self._bias
        return self._S

    def predict(
        self, X: npt.NDArray[np.float64], y: npt.ArrayLike | None = None
    ) -> npt.ArrayLike:
        """
        Predict the labels of the validation set.

        Args:
            X (npt.NDArray[np.float64]): The validation set.
            y (npt.ArrayLike, optional): The true labels of the validation set, defaults to None.

        Returns:
            npt.ArrayLike: The predicted labels of the validation set.
        """
        if not self._fitted:
            raise ValueError("Classifier has not been fitted yet.")

        self.scores(X)

        predictions = self._S > 0

        if y is not None:
            self.accuracy = np.mean(predictions == y) * 100
            self.error_rate = 100 - self.accuracy

        return predictions

    @property
    def llr(self):
        """
        Posterior Log likelihood ratio of the classifier.
        """

        return self._S - np.log(self._prior / (1 - self._prior))

    def fit(
        self, X, y, *, l: float, prior: float | None = None  # noqa: E741
    ) -> "LogisticRegression":
        """
        Fit the logistic regression classifier using the training data and the
        specified hyperparameters.

        Args:
            l (float): the regularization hyperparameter
            prior (float, optional): the prior probability of the positive class,
                if None, the standard logistic regression objective is used,
                otherwise the prior-weighted logistic regression objective is used

        Returns:
            float: the value of the objective function at the optimal point
        """

        if self._quadratic:
            X = quadratic_feature_expansion(X)

        log_reg = partial(self.objective, DTR=X, LTR=y, l=l, prior=prior)

        x, self._f, _ = opt.fmin_l_bfgs_b(log_reg, np.zeros(X.shape[0] + 1))

        self._weights, self._bias = x[:-1], x[-1]

        self._fitted = True
        self._prior = prior or np.mean(y)

        return self

    @staticmethod
    @njit(cache=True)
    def objective(
        v: npt.NDArray[np.float64],
        *,
        prior: float | None,
        DTR: npt.NDArray[np.float64],
        LTR: npt.NDArray[np.int64],
        l: float,  # noqa: E741
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

    def to_json(self, fp=None):
        if not self._fitted:
            raise ValueError("Classifier has not been fitted yet.")

        data = {
            "bias": self._bias,
            "prior": self._prior,
            "quadratic": self._quadratic,
            "weights": self._weights.tolist(),
        }

        if fp is None:
            return data

        json.dump(data, fp)

    @staticmethod
    def from_json(data):
        decoded = (
            json.load(data) if isinstance(data, TextIOWrapper) else json.loads(data)
        )

        cl = LogisticRegression(decoded["quadratic"])
        cl._bias = decoded["bias"]
        cl._prior = decoded["prior"]
        cl._weights = np.array(decoded["weights"])
        cl._fitted = True

        return cl
