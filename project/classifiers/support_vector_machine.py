import json
from functools import partial
from io import TextIOWrapper
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from numba import njit

from project.classifiers.classifier import Classifier
from project.funcs.base import vcol, vrow


class SupportVectorMachine(Classifier):
    """
    Support Vector Machine Classifier.

    Attributes:
        llr (npt.NDArray): The posterior log likelihood ratio of the classifier.
        accuracy (float): The accuracy of the classifier.
        error_rate (float): The error rate of the classifier.

        _bias (float): The bias term of the classifier.
        _C (float): The regularization hyperparameter.
        _degree (int): The degree parameter for the polynomial kernel.
        _fitted (bool): Whether the classifier has been fitted or not.
        _gamma (float): The gamma parameter for the RBF kernel.
        _K (int): The kernel parameter.
        _S (npt.NDArray): The scores of the classifier.
        _type (Literal["linear"] | Literal["poly_kernel"] | Literal["rbf_kernel"]):
            The type of SVM to use or kernel function to use for the kernel type.
        _weights (npt.NDArray): The weights of the classifier (for the linear SVM).
        _xi (float): The xi parameter, acts as a bias term for the non-linear SVM.
        _X_margin (npt.NDArray): The training data that is on the margin.
        _H_partial (npt.NDArray): The partial product of H for the labels converted
            in +1/-1 for scoring.
    """

    def __init__(
        self,
        type: (
            Literal["linear"] | Literal["poly_kernel"] | Literal["rbf_kernel"]
        ) = "linear",
    ) -> None:
        self._type = type

    @property
    def llr(self) -> npt.NDArray:
        if not hasattr(self, "_S"):
            raise ValueError("Scores have not been computed yet.")

        return self._S

    @staticmethod
    def from_json(data):
        decoded = (
            json.load(data) if isinstance(data, TextIOWrapper) else json.loads(data)
        )

        cl = SupportVectorMachine(decoded["type"])
        cl._C = decoded["C"]
        cl._K = decoded["K"]

        if cl._type == "linear":
            cl._weights = np.array(decoded["weights"])
            cl._bias = decoded["bias"]
        else:
            cl._H_partial = vcol(np.array(decoded["H_partial"]))
            cl._X_margin = np.array(decoded["X_margin"]).T
            cl._xi = decoded["xi"]

            if cl._type == "poly_kernel":
                cl._degree = decoded["degree"]
                cl._c = decoded["c"]
            else:
                cl._gamma = decoded["gamma"]

        cl._init_kernel_func()
        cl._fitted = True

        return cl

    @staticmethod
    @njit(cache=True)
    def poly_kernel(
        D1: npt.NDArray[np.float64],
        D2: npt.NDArray[np.float64],
        degree: float,
        c: float,
        xi: float,
    ):
        """
        Implementation of the polynomial kernel `k(xi, xj) = (xi^T xj + c)^degree`

        Args:
            D1 (npt.NDArray[np.float64]): The first point
            D2 (npt.NDArray[np.float64]): The second point
            degree (float): The degree of the polynomial
            c (float): The coefficient of the polynomial
            xi (float): The bias term

        Returns:
            npt.NDArray[np.float64]: The kernel matrix
        """
        return (np.dot(D1.T, D2) + c) ** degree + xi

    @staticmethod
    @njit(cache=True)
    def rbf_kernel(
        D1: npt.NDArray[np.float64],
        D2: npt.NDArray[np.float64],
        gamma: float,
        xi: float,
    ):
        """
        Implementation of the Gaussian Radial Basis Function kernel
        `k(xi, xj) = e^(-gamma ||xi - xj||^2)`

        Args:
            D1 (npt.NDArray[np.float64]): The first point
            D2 (npt.NDArray[np.float64]): The second point
            gamma (float): Defines the width of the kernel, small gamma -> wide
                kernel, large gamma -> narrow kernel
            xi (float): The bias term

        Returns:
            npt.NDArray[np.float64]: The kernel matrix
        """
        # Fast method to compute all pair-wise distances. Exploit the fact that
        # |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z) + xi

    @staticmethod
    @njit(cache=True)
    def __objective(
        H: npt.NDArray[np.float64], alpha: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        """
        Objective function for the SVM optimization problem
        `L = 1/2 alpha^T H alpha - 1^T alpha`
        `dL/dalpha = H alpha - 1`

        Args:
            H (npt.NDArray[np.float64]): The Hessian matrix
            alpha (npt.NDArray[np.float64]): The alpha values

        Returns:
            tuple[float, npt.NDArray[np.float64]]: The loss and the gradient
        """
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)

        return loss, grad

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray,
        *,
        C: float = 0,
        K: int = 1,
        xi: float = 1,
        degree: int = 1,
        c: float = 1,
        gamma: float = 1,
    ) -> "SupportVectorMachine":
        """
        Train the support vector machine classifier using the training data and
        the specified hyperparameters.

        Args:
            X (npt.NDArray[np.float64]): the training data.
            y (npt.NDArray): the training labels.

            C (float): the regularization hyperparameter.
            K (int, optional): hyperparameter that mitigates the regularization
                of the bias in linear kernels, for linear kernels we have that
                `xi = K^2` for non-linear ones. Defaults to 1.
            xi (float, optional): the xi parameter, acts as a bias term for the
                non-linear SVM. Defaults to 1.
            degree (int, optional): the degree parameter for the polynomial
                kernel. Defaults to 1.
            c (float, optional): the `c` constant for the polynomial kernel.
                Defaults to 1.
            gamma (float, optional): the gamma parameter for the RBF kernel.
                Defaults to 1.

        Returns:
            SupportVectorMachine: the fitted classifier.
        """

        # Hyperparameters
        self._C = C
        self._K = K
        self._xi = xi
        self._degree = degree
        self._c = c
        self._gamma = gamma

        self._init_kernel_func()

        ZTR = y * 2.0 - 1.0  # Convert labels to +1/-1

        if self._type == "linear":
            DTR_EXT = np.vstack([X, np.ones((1, X.shape[1])) * K])
            xixj = np.dot(DTR_EXT.T, DTR_EXT)
        else:
            xixj = self._kernel_func(X, X)  # type: ignore

        H = vcol(ZTR) * vrow(ZTR) * xixj

        alpha_star, *_ = opt.fmin_l_bfgs_b(
            partial(self.__objective, H),
            np.zeros(X.shape[1]),
            bounds=[(0, C) for _ in y],
            factr=1.0,
        )

        if self._type == "linear":
            # Compute primal solution for extended data matrix
            w_hat = (vrow(alpha_star) * vrow(ZTR) * DTR_EXT).sum(1)

            # b must be rescaled in case K != 1, since we want to compute w'x + b * K
            self._weights, self._bias = (w_hat[:-1], w_hat[-1] * K)
        else:
            # For non-linear SVM we need to save the training data and the
            # partial product of H for the labels converted in +1/-1 for scoring,
            # since points with alpha_star value of zero are outside the margin
            # and do not contribute to the decision boundary, we can filter
            # them out. NOTE: SVMs scale poorly with the quantity of data and
            # the training time increases quadratically with the number of
            # samples, so it would make sense to add a pre-processing step to
            # reduce the number of samples, that would also make the final
            # serialized model smaller
            non_zero_idx = np.nonzero(alpha_star)[0]
            self._H_partial = vcol(alpha_star[non_zero_idx]) * vcol(ZTR[non_zero_idx])
            self._X_margin = X[:, non_zero_idx]

        self._fitted = True

        return self

    def scores(self, X):
        if not self._fitted:
            raise ValueError("Classifier has not been fitted yet.")

        if self._type == "linear":
            self._S: npt.NDArray[np.float64] = (
                vrow(self._weights) @ X + self._bias
            ).ravel()
        else:
            xixj = self._kernel_func(self._X_margin, X)  # type: ignore
            H = self._H_partial * xixj
            self._S: npt.NDArray[np.float64] = H.sum(0)

        return self._S

    def to_json(self, fp=None):
        params = {}
        if self._type == "linear":
            params["weights"] = self._weights.tolist()
            params["bias"] = self._bias
        else:
            if self._type == "rbf_kernel":
                params["gamma"] = self._gamma
            else:
                params["degree"] = self._degree
                params["c"] = self._c

            params["xi"] = self._xi
            # Save as row to make the representation more compact
            params["H_partial"] = vrow(self._H_partial).tolist()
            # Transpose to make the data mimic our original one
            params["X_margin"] = self._X_margin.T.tolist()

        data = {
            "type": self._type,
            "C": self._C,
            "K": self._K,  # Even though it only really matters for linear SVMs
            **params,
        }

        if fp is None:
            return data

        json.dump(data, fp)

    def _init_kernel_func(self):
        """
        Initialize the kernel function based on the type of SVM being used as a
        partial function with the appropriate hyperparameters.
        """
        self._kernel_func = (
            (
                partial(self.poly_kernel, degree=self._degree, c=self._c, xi=self._xi)
                if self._type == "poly_kernel"
                else partial(self.rbf_kernel, gamma=self._gamma, xi=self._xi)
            )
            if self._type != "linear"
            else None
        )
