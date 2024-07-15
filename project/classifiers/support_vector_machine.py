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
    def __init__(
        self,
        svm_type: (
            Literal["linear"] | Literal["poly_kernel"] | Literal["rbf_kernel"]
        ) = "linear",
    ) -> None:
        """
        Initializes the support vector machine classifier.

        Args:
            svm_type (
                Literal[linear] | Literal[poly_kernel] | Literal[rbf_kernel],
                optional,
            ): the type of SVM to use or kernel function to use for the kernel
                type. Defaults to "linear".
        """
        self._svm_type = svm_type

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
        Train the support vector machine classifier using the training data and the
        specified hyperparameters.

        Args:
            X (npt.NDArray[np.float64]): the training data.
            y (npt.NDArray): the training labels.

            C (float): the regularization hyperparameter.
            K (int, optional): the kernel parameter. Defaults to 1.
            xi (float, optional): the xi parameter, acts as a bias term for the
                non-linear SVM. Defaults to 1.
            degree (int, optional): the degree parameter for the polynomial kernel.
                Defaults to 1.
            c (float, optional): the c parameter for the polynomial kernel. Defaults
                to 1.
            gamma (float, optional): the gamma parameter for the RBF kernel. Defaults
                to 1.

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

        self.__ZTR = y * 2.0 - 1.0  # Convert labels to +1/-1

        if self._svm_type == "linear":
            DTR_EXT = np.vstack([X, np.ones((1, X.shape[1])) * K])
            zizj = np.dot(DTR_EXT.T, DTR_EXT)
        else:
            zizj = self._kernel_func(X, X)  # type: ignore

        H = zizj * vcol(self.__ZTR) * vrow(self.__ZTR)

        self._alpha_star, *_ = opt.fmin_l_bfgs_b(
            partial(self.__objective, H),
            np.zeros(X.shape[1]),
            bounds=[(0, C) for _ in y],
            factr=1.0,
        )

        if self._svm_type == "linear":
            # Compute primal solution for extended data matrix
            w_hat = (vrow(self._alpha_star) * vrow(self.__ZTR) * DTR_EXT).sum(1)

            # b must be rescaled in case K != 1, since we want to compute w'x + b * K
            self._weights, self._bias = (w_hat[:-1], w_hat[-1] * K)

        self._fitted = True

        return self

    def _init_kernel_func(self):
        self._kernel_func = (
            (
                partial(self.poly_kernel, degree=self._degree, c=self._c, xi=self._xi)
                if self._svm_type == "poly_kernel"
                else partial(self.rbf_kernel, gamma=self._gamma, xi=self._xi)
            )
            if self._svm_type != "linear"
            else None
        )

    @property
    def llr(self) -> npt.NDArray:
        if not hasattr(self, "_S"):
            raise ValueError("Scores have not been computed yet.")

        return self._S

    def scores(
        self,
        X_val: npt.NDArray[np.float64],
        X_train: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        if not self._fitted:
            raise ValueError("Classifier has not been fitted yet.")

        if self._svm_type == "linear":
            self._S = (vrow(self._weights) @ X_val + self._bias).ravel()
        else:
            if X_train is None:
                raise ValueError("Training data must be provided for non-linear SVM.")

            zizj = self._kernel_func(X_train, X_val)  # type: ignore

            H = vcol(self._alpha_star) * vcol(self.__ZTR) * zizj

            self._S = H.sum(0)

        return self._S

    @staticmethod
    @njit(cache=True)
    def __objective(H: npt.NDArray[np.float64], alpha: npt.NDArray[np.float64]):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

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
        Implementation of the polynomial kernel

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

    def to_json(self, fp=None):
        params = (
            {
                "weights": self._weights.tolist(),
                "bias": self._bias,
            }
            if self._svm_type == "linear"
            else (
                {
                    "xi": self._xi,
                    "gamma": self._gamma,
                    "alpha_star": self._alpha_star.tolist(),
                }
                if self._svm_type == "rbf_kernel"
                else (
                    {
                        "degree": self._degree,
                        "c": self._c,
                        "xi": self._xi,
                        "alpha_star": self._alpha_star.tolist(),
                    }
                    if self._svm_type == "poly_kernel"
                    else {}
                )
            )
        )

        data = {
            "svm_type": self._svm_type,
            "C": self._C,
            "K": self._K,
            "ZTR": self.__ZTR.tolist(),
            **params,
        }

        if fp is None:
            return data

        json.dump(data, fp)

    @staticmethod
    def from_json(data):
        decoded = (
            json.load(data) if isinstance(data, TextIOWrapper) else json.loads(data)
        )

        cl = SupportVectorMachine(decoded["svm_type"])
        cl._C = decoded["C"]
        cl._K = decoded["K"]
        cl.__ZTR = np.array(decoded["ZTR"])

        if cl._svm_type == "linear":
            cl._weights = np.array(decoded["weights"])
            cl._bias = decoded["bias"]
        elif cl._svm_type == "rbf_kernel":
            cl._alpha_star = np.array(decoded["alpha_star"])
            cl._xi = decoded["xi"]
            cl._gamma = decoded["gamma"]
        else:
            cl._alpha_star = np.array(decoded["alpha_star"])
            cl._degree = decoded["degree"]
            cl._c = decoded["c"]
            cl._xi = decoded["xi"]

        cl._init_kernel_func()
        cl._fitted = True

        return cl
