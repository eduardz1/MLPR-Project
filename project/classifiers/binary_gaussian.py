import json
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Literal

import numpy as np
import numpy.typing as npt

from project.classifiers.classifier import Classifier
from project.funcs.base import corr, cov, vcol
from project.funcs.dcf import optimal_bayes_threshold
from project.funcs.log_pdf import log_pdf_gaussian
from project.funcs.pca import pca


@dataclass
class BinaryGaussian(Classifier):
    """
    Attributes:
        type (Literal["naive", "tied", "multivariate"]): Type of classifier to use.

        C (list[npt.NDArray]): Covariance matrices of the classes.
        mu (list[npt.NDArray]): Means of the classes.
        corr (list[npt.NDArray]): Correlation matrices of the classes.
        llr (npt.NDArray): Log likelihood ratio of the classifier.
        accuracy (float): Accuracy of the classifier.
        error_rate (float): Error rate of the classifier.

        _S (npt.NDArray): Scores of the classifier.
        _pca_dims (int): Number of dimensions to keep after PCA.
        _slicer (slice): Slice to apply to the data.
        _fitted (bool): Whether the classifier has been fitted or not.
    """

    def __init__(self, classifier: Literal["naive", "tied", "multivariate"]) -> None:
        self.type = classifier
        self._fitted = False

    def scores(self, X):
        if not self._fitted:
            raise ValueError("Classifier has not been fitted yet.")

        if self._slicer:
            X = X[:, self._slicer]

        if self._pca_dims:
            X = pca(X, self._pca_dims)[1]

        X = X.T

        self._S = np.zeros((2, X.shape[1]))
        for i in [0, 1]:
            mu = self.mu[i]
            C = self.C[i]
            self._S[i, :] = log_pdf_gaussian(X, vcol(mu), C)

        return self._S

    @property
    def llr(self):
        """
        Log likelihood ratio of the classifier. llr(xₜ) = log(𝑓(xₜ|h₁) / 𝑓(xₜ|h₀))
        """
        if not hasattr(self, "_S"):
            raise ValueError("Scores have not been computed yet.")

        return self._S[1] - self._S[0]

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.ArrayLike,
        *,
        slicer: slice | None = None,
        pca_dims: int | None = None,
    ) -> "BinaryGaussian":
        """
        Fit the Gaussian classifier to the training data.

        Args:
            X (npt.NDArray[np.float64]): Training data.
            y (npt.ArrayLike): Training labels.
            slicer (slice, optional): Slice to apply to the data. Defaults to None.
            pca_dims (int, optional): Number of dimensions to keep after PCA
                if one want to apply it as a pre-processing step. Defaults to None.

        Returns:
            BinaryGaussian: The fitted classifier.
        """

        self._pca_dims = pca_dims
        self._slicer = slicer

        X = X[:, slicer] if slicer else X

        if pca_dims:
            X = pca(X, pca_dims)[1]

        split = [X[y == k] for k in [0, 1]]

        self.C = [cov(split[k].T) for k in [0, 1]]
        self.mu = [np.mean(split[k], axis=0) for k in [0, 1]]
        self.corr = [corr(split[k].T) for k in [0, 1]]

        if self.type == "tied":
            Sw = np.average(  # Within-class covariance matrix
                [self.C[k] for k in [0, 1]],
                axis=0,
                weights=np.array([len(split[k].T) for k in [0, 1]]),
            )

            # If tied then the ML estimate of the covariance matrix is
            # the within class covariance matrix
            self.C = [Sw, Sw]
        elif self.type == "naive":
            # If naive then the ML estimate of the covariance matrix is
            # the diagonal of the sample covariance matrix
            self.C = [np.diag(np.diag(self.C[k])) for k in [0, 1]]

        self._fitted = True

        return self

    def predict(
        self,
        X: npt.NDArray[np.float64],
        y: npt.ArrayLike | None = None,
        *,
        pi_T: float = 0.5,
        C_fn: float = 1,
        C_fp: float = 1,
    ) -> npt.ArrayLike:
        """
        Predict the class of the samples in the validation set.

        Args:
            X (npt.NDArray[np.float64]): Validation set.
            y (npt.ArrayLike, optional): True labels of the validation set, if
                provided the accuracy and error rate will be computed. Defaults to None.
            pi_T (float, optional): Prior of the True class. Defaults to 0.5.
            C_fn (float, optional): Cost of false negatives. Defaults to 1.
            C_fp (float, optional): Cost of false positives. Defaults to 1.

        Returns:
            ArrayLike: Predicted classes of the samples in the validation set.
        """
        if not self._fitted:
            raise ValueError("Classifier has not been fitted yet.")

        self.scores(X)

        predictions = self.llr > optimal_bayes_threshold(pi_T, C_fn, C_fp)

        if y is not None:
            self.accuracy = np.mean(predictions == y) * 100
            self.error_rate = 100 - self.accuracy

        return predictions

    @staticmethod
    def from_json(data):
        decoded = (
            json.load(data) if isinstance(data, TextIOWrapper) else json.loads(data)
        )

        cl = BinaryGaussian(decoded["classifier"])
        cl.mu = decoded["mu"]
        cl.C = decoded["C"]
        cl._pca_dims = decoded["pca_dims"]
        cl._slicer = decoded["slicer"]
        cl._fitted = True

        return cl

    def to_json(self, fp=None):
        if not self._fitted:
            raise ValueError("Classifier has not been fitted yet.")

        data = {
            "classifier": self.type,
            "mu": [mu.tolist() for mu in self.mu],
            "C": [C.tolist() for C in self.C],
            "pca_dims": self._pca_dims,
            "slicer": self._slicer,
        }

        if fp is None:
            return data

        json.dump(data, fp)
