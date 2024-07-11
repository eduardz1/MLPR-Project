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

    def train(
        self,
        C: float,
        svm_type: Literal["linear"] | Literal["kernel"] = "linear",
        K: int = 1,
        eps: float = 1,
        kernel_func: Literal["poly_kernel"] | Literal["rbf_kernel"] | None = None,
        degree: int = 1,
        c: float = 1,
        gamma: float = 1,
    ) -> None:
        """
        Train the support vector machine classifier using the training data and the
        specified hyperparameters.

        Args:
            C (float): the regularization hyperparameter
            svm_type (Literal[linear] | Literal[kernel], optional): the type of
                SVM to use. Defaults to "linear".
            K (int, optional): the kernel parameter. Defaults to 1.
            eps (float, optional): the epsilon parameter. Defaults to 1.
            kernel_func (Literal[poly_kernel] | Literal[rbf_kernel], optional): the
                kernel function to use. Defaults to None.
            degree (int, optional): the degree parameter for the polynomial kernel.
                Defaults to 1.
            c (float, optional): the c parameter for the polynomial kernel. Defaults
                to 1.
            gamma (float, optional): the gamma parameter for the RBF kernel. Defaults
                to 1.
        """

        self._svm_type = svm_type
        self._kernel_func = kernel_func
        self._eps = eps
        self._degree = degree
        self._c = c
        self._gamma = gamma

        ZTR = self.y_train * 2.0 - 1.0  # Convert labels to +1/-1

        if svm_type == "linear":
            DTR_EXT = np.vstack([self.X_train, np.ones((1, self.X_train.shape[1])) * K])
            zizj = np.dot(DTR_EXT.T, DTR_EXT)
        else:
            zizj = (
                self.poly_kernel(self.X_train, self.X_train, degree, c)
                if kernel_func == "poly_kernel"
                else self.rbf_kernel(self.X_train, self.X_train, gamma)
            ) + eps

        H = zizj * vcol(ZTR) * vrow(ZTR)

        self._alpha_star, *_ = opt.fmin_l_bfgs_b(
            partial(self.__objective, H),
            np.zeros(self.X_train.shape[1]),
            bounds=[(0, C) for _ in self.y_train],
            factr=1.0,
        )

        if svm_type == "linear":
            # Compute primal solution for extended data matrix
            w_hat = (vrow(self._alpha_star) * vrow(ZTR) * DTR_EXT).sum(1)

            # b must be rescaled in case K != 1, since we want to compute w'x + b * K
            self._weights, self._bias = (w_hat[:-1], w_hat[-1] * K)

    @property
    def llr(self) -> npt.NDArray:
        """
        Log-likelihood ratio of the classifier.
        """

        if self._svm_type == "linear":
            return (vrow(self._weights) @ self.X_val + self._bias).ravel()
        else:
            ZTR = self.y_train * 2.0 - 1.0  # Convert labels to +1/-1
            zizj = (
                self.poly_kernel(self.X_train, self.X_val, self._degree, self._c)
                if self._kernel_func == "poly_kernel"
                else self.rbf_kernel(self.X_train, self.X_val, self._gamma)
            ) + self._eps

            H = vcol(self._alpha_star) * vcol(ZTR) * zizj

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
    ):
        return (np.dot(D1.T, D2) + c) ** degree

    @staticmethod
    @njit(cache=True)
    def rbf_kernel(
        D1: npt.NDArray[np.float64], D2: npt.NDArray[np.float64], gamma: float
    ):
        """
        Implementation of the Gaussian Radial Basis Function kernel

        Args:
            D1 (npt.NDArray[np.float64]): The first point
            D2 (npt.NDArray[np.float64]): The second point
            gamma (float): Defines the width of the kernel, small gamma -> wide
                kernel, large gamme -> narrow kernel

        Returns:
            _type_: _description_
        """
        # Fast method to compute all pair-wise distances. Exploit the fact that
        # |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)

    def to_json(self) -> dict:
        params = (
            {
                "weights": self._weights.tolist(),
                "bias": self._bias,
            }
            if self._svm_type == "linear"
            else {
                "alpha_star": self._alpha_star.tolist(),
                "eps": self._eps,
                "kernel_func": self._kernel_func,
                "degree": self._degree,
                "c": self._c,
                "gamma": self._gamma,
            }
        )

        return {
            "svm_type": self._svm_type,
            **params,
        }

    @staticmethod
    def from_json(data: dict) -> "SupportVectorMachine":
        svm = SupportVectorMachine.__new__(SupportVectorMachine)
        svm._svm_type = data["svm_type"]

        if svm._svm_type == "linear":
            svm._weights = data["weights"]
            svm._bias = data["bias"]
        else:
            svm._alpha_star = data["alpha_star"]
            svm._eps = data["eps"]
            svm._kernel_func = data["kernel_func"]
            svm._degree = data["degree"]
            svm._c = data["c"]
            svm._gamma = data["gamma"]

        return svm
