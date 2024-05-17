from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split

from project.funcs.logpdf import log_pdf
from project.funcs.pca import pca


class Gaussian:
    def __init__(self, X: ArrayLike, y: ArrayLike):
        self.X = X
        self.y = y
        self.classes = np.unique(y)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.33, random_state=0
        )

        self._X_train = X_train.T
        self._X_val = X_val.T
        self._y_train = y_train
        self._y_val = y_val

        self.covariances = {
            k: np.cov(self._X_train[:, self._y_train == k], bias=True)
            for k in self.classes
        }
        self.means = {
            k: np.mean(self._X_train[:, self._y_train == k], axis=1, keepdims=True)
            for k in self.classes
        }
        self.corrcoef = {
            k: np.corrcoef(self._X_train[:, self._y_train == k]) for k in self.classes
        }

    def fit(self, **kwargs) -> tuple[float, float, Optional[ArrayLike]]:
        """
        Fit the Gaussian classifier to the training data and return the accuracy
        and error rate on the validation data.

        Returns:
            tuple[float, float, Optional[ArrayLike]]: Accuracy and error rate
                in percentage and log-likelihood ratio (LLR)
        """

        slice = kwargs.get("slice", None)

        X_train = self._X_train if slice is None else self._X_train[slice]
        X_val = self._X_val if slice is None else self._X_val[slice]

        X_train = (
            pca(X_train.T, kwargs.get("pca_dimensions", 2))[1].T
            if kwargs.get("pca")
            else X_train
        )
        X_val = (
            pca(X_val.T, kwargs.get("pca_dimensions", 2))[1].T
            if kwargs.get("pca")
            else X_val
        )

        self.covariances = {
            k: np.cov(X_train[:, self._y_train == k], bias=True) for k in self.classes
        }
        self.means = {
            k: np.mean(X_train[:, self._y_train == k], axis=1, keepdims=True)
            for k in self.classes
        }

        if kwargs.get("tied"):
            return (*self._tied_covariance(X_train, X_val), None)
        elif kwargs.get("naive"):
            return (*self._naive(X_val), None)
        else:
            return self._multivariate(X_val)

    def _multivariate(self, X_val) -> tuple[float, float, ArrayLike]:
        """
        Returns:
            tuple[float, float, ArrayLike]: Accuracy and error rate in
                percentage and log-likelihood ratio  (LLR)
        """

        S = np.zeros((len(self.classes), X_val.shape[1]))
        for i, c in enumerate(self.classes):
            mu = self.means[c]
            C = self.covariances[c]

            log_pdf_val = log_pdf(X_val, mu, C)

            likelihood = np.exp(log_pdf_val)

            S[i, :] = likelihood

        llr = S[1] / S[0]

        # We assume uniform class priors
        SJoint = S * 0.5
        SMarginal = SJoint.sum(axis=0)
        SPost = SJoint / SMarginal

        y_pred = np.argmax(SPost, axis=0)

        accuracy = np.sum(self._y_val == y_pred) / self._y_val.size * 100
        error_rate = 100 - accuracy

        return accuracy, error_rate, llr

    def _tied_covariance(self, X_train, X_val) -> tuple[float, float]:
        """
        Returns:
            tuple[float, float]: Accuracy and error rate in percentage
        """

        weights = np.array([len(X_train[:, self._y_train == c]) for c in self.classes])
        Sw = np.average(
            [self.covariances[c] for c in self.classes],
            axis=0,
            weights=weights,
        )

        S = np.zeros((len(self.classes), X_val.shape[1]))
        for i, c in enumerate(self.classes):
            mu = self.means[c]

            log_pdf_val = log_pdf(X_val, mu, Sw)

            likelihood = np.exp(log_pdf_val)

            S[i, :] = likelihood

        SJoint = S * 0.5
        SMarginal = SJoint.sum(axis=0)
        SPost = SJoint / SMarginal

        y_pred = np.argmax(SPost, axis=0)

        accuracy = np.sum(self._y_val == y_pred) / self._y_val.size * 100
        error_rate = 100 - accuracy

        return accuracy, error_rate

    def _naive(self, X_val) -> tuple[float, float]:
        """
        Returns:
            tuple[float, float]: Accuracy and error rate in percentage
        """

        S = np.zeros((len(self.classes), X_val.shape[1]))
        for i, c in enumerate(self.classes):
            mu = self.means[c]
            C = np.atleast_2d(self.covariances[c])

            # Diagonalize the covariance matrix
            C = np.diag(np.diag(C))

            log_pdf_val = log_pdf(X_val, mu, C)

            likelihood = np.exp(log_pdf_val)

            S[i, :] = likelihood

        SJoint = S * 0.5
        SMarginal = SJoint.sum(axis=0)
        SPost = SJoint / SMarginal

        y_pred = np.argmax(SPost, axis=0)

        accuracy = np.sum(self._y_val == y_pred) / self._y_val.size * 100
        error_rate = 100 - accuracy

        return accuracy, error_rate
