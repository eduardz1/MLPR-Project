from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from numba import njit

from project.funcs.base import vcol, vrow


@dataclass
class SupportVectorMachine:
    X_train: npt.NDArray
    y_train: npt.NDArray
    X_val: npt.NDArray
    y_val: npt.NDArray

    def train(
        self,
        svm_type: (
            Literal["linear"] | Literal["poly_kernel"] | Literal["rbf_kernel"]
        ) = "linear",
        *,
        C: float = 0,
        K: int = 1,
        xi: float = 1,
        degree: int = 1,
        c: float = 1,
        gamma: float = 1,
    ) -> None:
        """
        Train the support vector machine classifier using the training data and the
        specified hyperparameters.

        Args:
            C (float): the regularization hyperparameter
            svm_type (
                Literal[linear] | Literal[poly_kernel] | Literal[rbf_kernel],
                optional,
            ): the type of SVM to use or kernel function to use for the kernel
                type. Defaults to "linear".
            K (int, optional): the kernel parameter. Defaults to 1.
            xi (float, optional): the xi parameter, acts as a bias term for the
                non-linear SVM. Defaults to 1.
            degree (int, optional): the degree parameter for the polynomial kernel.
                Defaults to 1.
            c (float, optional): the c parameter for the polynomial kernel. Defaults
                to 1.
            gamma (float, optional): the gamma parameter for the RBF kernel. Defaults
                to 1.
        """

        self._svm_type = svm_type
        self._init_kernel_func()

        # Hyperparameters
        self._C = C
        self._K = K
        self._xi = xi
        self._degree = degree
        self._c = c
        self._gamma = gamma

        self.__ZTR = self.y_train * 2.0 - 1.0  # Convert labels to +1/-1

        if svm_type == "linear":
            DTR_EXT = np.vstack([self.X_train, np.ones((1, self.X_train.shape[1])) * K])
            zizj = np.dot(DTR_EXT.T, DTR_EXT)
        else:
            zizj = self._kernel_func(self.X_train, self.X_train)  # type: ignore

        H = zizj * vcol(self.__ZTR) * vrow(self.__ZTR)

        self._alpha_star, *_ = opt.fmin_l_bfgs_b(
            partial(self.__objective, H),
            np.zeros(self.X_train.shape[1]),
            bounds=[(0, C) for _ in self.y_train],
            factr=1.0,
        )

        if svm_type == "linear":
            # Compute primal solution for extended data matrix
            w_hat = (vrow(self._alpha_star) * vrow(self.__ZTR) * DTR_EXT).sum(1)

            # b must be rescaled in case K != 1, since we want to compute w'x + b * K
            self._weights, self._bias = (w_hat[:-1], w_hat[-1] * K)

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
        return self.scores

    @property
    def scores(self) -> npt.NDArray:
        """
        LLR-like measure of the classifier (NOTE: not a real log-likelihood ratio)
        """
        if self._svm_type == "linear":
            return (vrow(self._weights) @ self.X_val + self._bias).ravel()
        else:
            zizj = self._kernel_func(self.X_train, self.X_val)  # type: ignore

            H = vcol(self._alpha_star) * vcol(self.__ZTR) * zizj

            return H.sum(0)

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

    def to_json(self) -> dict:
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

        return {
            "svm_type": self._svm_type,
            "C": self._C,
            "K": self._K,
            "ZTR": self.__ZTR.tolist(),
            **params,
        }

    @staticmethod
    def from_json(data: dict) -> "SupportVectorMachine":
        svm = SupportVectorMachine.__new__(SupportVectorMachine)
        svm._svm_type = data["svm_type"]
        svm._C = data["C"]
        svm._K = data["K"]
        svm.__ZTR = np.array(data["ZTR"])

        if svm._svm_type == "linear":
            svm._weights = np.array(data["weights"])
            svm._bias = data["bias"]
        elif svm._svm_type == "rbf_kernel":
            svm._alpha_star = np.array(data["alpha_star"])
            svm._xi = data["xi"]
            svm._gamma = data["gamma"]
        else:
            svm._alpha_star = np.array(data["alpha_star"])
            svm._degree = data["degree"]
            svm._c = data["c"]
            svm._xi = data["xi"]

        svm._init_kernel_func()

        return svm
