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
        eps: float = 1,
        kernel_func: Callable[[npt.NDArray, npt.NDArray], npt.NDArray] | None = None,
    ):
        """
        Train the support vector machine classifier using the training data and the
        specified hyperparameters.

        Args:
            C (float): the regularization hyperparameter
            svm_type (Literal[linear] | Literal[kernel], optional): the type of
                SVM to use. Defaults to "linear".
            K (int, optional): the kernel parameter. Defaults to 1.
            eps (float, optional): the epsilon parameter. Defaults to 1.
            kernel_func (
                Callable[[npt.NDArray, npt.NDArray], npt.NDArray] | None, optional
                ): the kernel function to use. Defaults to None.

        Raises:
            ValueError: Kernel function must be provided when using kernel SVM

        Returns:
            npt.NDArray: the scores of the classifier on the validation data
        """

        if svm_type == "kernel" and kernel_func is None:
            raise ValueError("Kernel function must be provided when using kernel SVM")

        ZTR = self.y_train * 2.0 - 1.0  # Convert labels to +1/-1

        if svm_type == "linear":
            DTR_EXT = np.vstack([self.X_train, np.ones((1, self.X_train.shape[1])) * K])
            ker = np.dot(DTR_EXT.T, DTR_EXT)
        else:
            ker = kernel_func(self.X_train, self.X_train) + eps  # type: ignore

        H = ker * vcol(ZTR) * vrow(ZTR)

        # Dual objective with gradient
        fopt = partial(self.__function_optimize, H)

        alpha_star, *_ = opt.fmin_l_bfgs_b(
            fopt,
            np.zeros(self.X_train.shape[1]),
            bounds=[(0, C) for _ in self.y_train],
            factr=1.0,
        )

        if svm_type == "linear":
            # Compute primal solution for extended data matrix
            w_hat = (vrow(alpha_star) * vrow(ZTR) * DTR_EXT).sum(1)

            # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
            w, b = (
                w_hat[0 : self.X_train.shape[0]],
                w_hat[-1] * K,
            )  # b must be rescaled in case K != 1, since we want to compute w'x + b * K

            self.primal_loss = self.__calculate_primal_loss(w_hat, DTR_EXT, ZTR, C)
            self.dual_loss = -self.__function_optimize(H, alpha_star)[0]
            self.duality_gap = self.primal_loss - self.dual_loss

            self.scores = (vrow(w) @ self.X_val + b).ravel()
        else:
            self.dual_loss = -self.__function_optimize(H, alpha_star)[0]

            ker = kernel_func(self.X_train, self.X_val) + eps  # type: ignore
            H = vcol(alpha_star) * vcol(ZTR) * ker
            self.scores = H.sum(0)  # compute the fscore

        return self.scores

    def __function_optimize(self, H, alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    def __calculate_primal_loss(self, w_hat, DTR_EXT, ZTR, C):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * np.linalg.norm(w_hat) ** 2 + C * np.maximum(0, 1 - ZTR * S).sum()
