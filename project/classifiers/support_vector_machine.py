import numpy as np
import scipy

from project.funcs.base import vcol, vrow


class SupportVectorMachine:
    def __init__(self, DTR, LTR):
        self.DTR = DTR
        self.LTR = LTR

    def train(self, C, svm_type="linear", K=1, kernelFunc=None):
        """
        Train a SVM over a dataset
        Args:
            C:  hyperparameter.
            svm_type - optional: is either 'linear' or 'kernel'. Default to 'linear'
                If linear, then the returned values are the weights w and the bias b.
                If kernel, then the returned values is the score function fscore.
                For the kernel, a kernel function is needed to compute the kernel matrix.
            K - optional: regularization term. Default is 1.
            kernelFunc - optional: kernel function. Default is None.
        Returns: w, b if svm_type is 'linear', a function fscore if svm_type is 'kernel'
        """
        if svm_type == "linear":
            return self.__train_dual_SVM_linear(C, K)
        if svm_type == "kernel":
            return self.__train_dual_SVM_kernel(C, kernelFunc, K)

    def __train_dual_SVM_linear(self, C, K=1):

        ZTR = self.LTR * 2.0 - 1.0  # Convert labels to +1/-1
        DTR_EXT = np.vstack([self.DTR, np.ones((1, self.DTR.shape[1])) * K])

        H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

        # Dual objective with gradient
        def fOpt(alpha):
            Ha = H @ vcol(alpha)
            loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
            grad = Ha.ravel() - np.ones(alpha.size)
            return loss, grad

        alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
            fOpt,
            np.zeros(DTR_EXT.shape[1]),
            bounds=[(0, C) for i in self.LTR],
            factr=1.0,
        )

        # Primal loss
        def primalLoss(w_hat):
            S = (vrow(w_hat) @ DTR_EXT).ravel()
            return (
                0.5 * np.linalg.norm(w_hat) ** 2 + C * np.maximum(0, 1 - ZTR * S).sum()
            )

        # Compute primal solution for extended data matrix
        w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)

        # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
        w, b = (
            w_hat[0 : self.DTR.shape[0]],
            w_hat[-1] * K,
        )  # b must be rescaled in case K != 1, since we want to compute w'x + b * K

        primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
        print(
            "SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e"
            % (C, K, primalLoss, dualLoss, primalLoss - dualLoss)
        )

        return w, b

    # kernelFunc: function that computes the kernel matrix from two data matrices
    def __train_dual_SVM_kernel(self, C, kernelFunc, eps=1.0):

        ZTR = self.LTR * 2.0 - 1.0  # Convert labels to +1/-1

        K = kernelFunc(self.DTR, self.DTR) + eps
        H = vcol(ZTR) * vrow(ZTR) * K

        # Dual objective with gradient
        def fOpt(alpha):
            Ha = H @ vcol(alpha)
            loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
            grad = Ha.ravel() - np.ones(alpha.size)
            return loss, grad

        alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
            fOpt,
            np.zeros(self.DTR.shape[1]),
            bounds=[(0, C) for i in self.LTR],
            factr=1.0,
        )

        print("SVM (kernel) - C %e - dual loss %e" % (C, -fOpt(alphaStar)[0]))

        def fScore(DTE):
            """
            Computes the SVM score for a matrix of test samples(DTE)
            Args:
                DTE: matrix of test samples
            """

            K = kernelFunc(self.DTR, DTE) + eps
            H = vcol(alphaStar) * vcol(ZTR) * K
            return H.sum(0)

        return fScore  # we directly return the function to score a matrix of test samples DTE
