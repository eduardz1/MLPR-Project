from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt

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
        kernelFunc: Callable[[npt.NDArray, npt.NDArray], npt.NDArray] | None = None,
    ):
        if svm_type == "linear":
            return self.__train_dual_SVM_linear(C, K)

        return self.__train_dual_SVM_kernel(C, kernelFunc, K)

    def __train_dual_SVM_linear(self, C: float, K: int = 1):
        ZTR = self.y_train * 2.0 - 1.0  # Convert labels to +1/-1
        DTR_EXT = np.vstack([self.X_train, np.ones((1, self.X_train.shape[1])) * K])

        H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

        # Dual objective with gradient

        fopt = partial(self.__function_optimize, H)

        alphaStar, _, _ = opt.fmin_l_bfgs_b(
            fopt,
            np.zeros(DTR_EXT.shape[1]),
            bounds=[(0, C) for i in self.y_train],
            factr=1.0,
        )

        # Compute primal solution for extended data matrix
        w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)

        # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
        w, b = (
            w_hat[0 : self.X_train.shape[0]],
            w_hat[-1] * K,
        )  # b must be rescaled in case K != 1, since we want to compute w'x + b * K

        primalLoss, dualLoss = (
            self.__calculate_primal_loss(w_hat, DTR_EXT, ZTR, C),
            -self.__function_optimize(H, alphaStar)[0],
        )
        print(
            "SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e"
            % (C, K, primalLoss, dualLoss, primalLoss - dualLoss)
        )

        self.scores = (vrow(w) @ self.X_val + b).ravel()

        return self.scores

    # kernelFunc: function that computes the kernel matrix from two data matrices
    def __train_dual_SVM_kernel(self, C, kernelFunc, eps=1.0):

        ZTR = self.y_train * 2.0 - 1.0  # Convert labels to +1/-1

        K = kernelFunc(self.X_train, self.X_train) + eps
        H = vcol(ZTR) * vrow(ZTR) * K

        # Dual objective with gradient
        fopt = partial(self.__function_optimize, H)

        alphaStar, _, _ = opt.fmin_l_bfgs_b(
            fopt,
            np.zeros(self.X_train.shape[1]),
            bounds=[(0, C) for i in self.y_train],
            factr=1.0,
        )

        print(
            "SVM (kernel) - C %e - dual loss %e"
            % (C, -self.__function_optimize(H, alphaStar)[0])
        )

        # Compute the fscore
        K = kernelFunc(self.X_train, self.X_val) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        self.scores = H.sum(0)

        return self.scores

    def __function_optimize(self, H, alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    def __calculate_primal_loss(self, w_hat, DTR_EXT, ZTR, C):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * np.linalg.norm(w_hat) ** 2 + C * np.maximum(0, 1 - ZTR * S).sum()
