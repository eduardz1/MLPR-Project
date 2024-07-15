from functools import cache, cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from project.funcs.base import corr, cov
from project.funcs.log_pdf import log_pdf_gaussian
from project.funcs.pca import pca


class BinaryGaussian:
    """
    Creates a binary Gaussian classifier that can be fitted to the input data
    with different strategies and configurations.

    Attributes:
        X_train (npt.NDArray): Training data.
        y_train (npt.NDArray): Training labels.
        X_val (npt.NDArray): Validation data.
        y_val (npt.NDArray): Validation labels.
    """

    def __init__(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_val: npt.NDArray,
        y_val: npt.NDArray,
    ) -> None:
        self.__X_train = X_train
        self.__y_train = y_train
        self.__X_val = X_val
        self.__y_val = y_val

        # Score matrix containing the class-conditional probabilities
        self.__S = np.zeros((2, self.__X_val.shape[1]))

    @cached_property
    def covariances(self) -> list[npt.NDArray]:
        """
        List of the covariance matrices of the two classes.
        C = (1/N) * Σ(x - μ)(x - μ)ᵀ
        """
        return [cov(self.__X_train[:, self.__y_train == k]) for k in [0, 1]]

    @cached_property
    def means(self) -> list[npt.NDArray]:
        """
        List of the means of the two classes. μ = (1/N) * Σx
        """
        return [
            np.mean(self.__X_train[:, self.__y_train == k], axis=1, keepdims=True)
            for k in [0, 1]
        ]

    @cached_property
    def corrcoef(self) -> list[npt.NDArray]:
        """
        List of the correlation matrices of the two classes.
        Corr(Xᵢ, Xⱼ) = C(Xᵢ, Xⱼ) / (σᵢ * σⱼ)
        """
        return [corr(self.__X_train[:, self.__y_train == k]) for k in [0, 1]]

    @property
    def accuracy(self) -> float:
        """
        Accuracy measure of the classifier.
        """
        return np.sum(self.__y_val == self.predict()) / self.__y_val.size * 100

    @property
    def error_rate(self) -> float:
        """
        Error rate measure of the classifier.
        """
        return 100 - self.accuracy

    @property
    def llr(self) -> npt.NDArray:
        """
        Log likelihood ratio of the classifier. llr(xₜ) = log(𝑓(xₜ|h₁) / 𝑓(xₜ|h₀))
        """
        return self.__S[1] - self.__S[0]

    def fit(
        self,
        *,
        slicer: slice | None = None,
        pca_dimensions: int | None = None,
        classifier: (
            Literal["naive"] | Literal["tied"] | Literal["multivariate"]
        ) = "multivariate"
    ) -> None:
        """
        Fit the Gaussian classifier to the training data. Defaults to
        multivariate gaussian if neither tied nor naive are specified.

        Args:
            slicer (slice, optional): Slice to apply to the data. Defaults to None.
            pca_dimensions (int, optional): Number of dimensions to keep after PCA
                if one want to apply it as a pre-processing step. Defaults to None.
            classifier ("naive" | "tied" | "multivariate", optional): Type of
                classifier to use. Defaults to "multivariate".
        """

        self.__classifier = classifier
        self.__pca_dimensions = pca_dimensions

        # Slice the data first if needed
        X_train = self.__X_train[slicer] if slicer else self.__X_train
        X_val = self.__X_val[slicer] if slicer else self.__X_val

        # Apply PCA to the data if needed
        if pca_dimensions:
            X_train = pca(X_train.T, pca_dimensions)[1].T
            X_val = pca(X_val.T, pca_dimensions)[1].T

        # Recompute the ML estimates if the data has been transformed either by
        # PCA or by slicing
        if (slicer or pca_dimensions) is not None:
            covariances = [cov(X_train[:, self.__y_train == k]) for k in [0, 1]]
            means = [
                np.mean(X_train[:, self.__y_train == k], axis=1, keepdims=True)
                for k in [0, 1]
            ]
        else:
            covariances = self.covariances
            means = self.means

        # Compute the within class covariance matrix if the `tied covariance`
        # classifier is selected
        if classifier == "tied":
            Sw = np.average(
                [covariances[c] for c in [0, 1]],
                axis=0,
                weights=np.array(
                    [len(X_train[:, self.__y_train == c]) for c in [0, 1]]
                ),
            )

        # Compute the log likelihoods for each sample of the validation set
        for i in [0, 1]:
            mu = means[i]

            # fmt: off
            C = (
                # If naive then the ML estimate of the covariance matrix is
                # the diagonal of the sample covariance matrix
                np.diag(np.diag(covariances[i]))
                if classifier == "naive"

                # If tied then the ML estimate of the covariance matrix is
                # the within class covariance matrix
                else Sw
                if classifier == "tied"

                # Otherwise, for the multivariate case, the ML estimate of the
                # covariance matrix is the sample covariance matrix
                else covariances[i]
            )
            # fmt: on

            # Store the log likelihoods for each sample of class i
            self.__S[i, :] = log_pdf_gaussian(X_val, mu, C)

    def predict(self, pi_T=0.5, C_fn=1, C_fp=1) -> npt.ArrayLike:
        """
        Predict the class of the samples in the validation set.

        Args:
            pi_T (float, optional): Prior of the True class. Defaults to 0.5.
            C_fn (float, optional): Cost of false negatives. Defaults to 1.
            C_fp (float, optional): Cost of false positives. Defaults to 1.

        Returns:
            ArrayLike: Predicted classes of the samples in the validation set.
        """
        return self.llr > self._optimal_threshold(pi_T, C_fn, C_fp)

    @cache
    def _optimal_threshold(self, pi_T: float, C_fn: float, C_fp: float) -> npt.NDArray:
        """
        Args:
            pi_T (float): prior of the True class.
            C_fn (float): Cost of false negatives.
            C_fp (float): Cost of false positives.

        Returns:
            ndarray: -log((πᴛ * C𝑓𝑛) / ((1 - πᴛ) * C𝑓𝑝))
        """
        return -np.log((pi_T * C_fn) / ((1 - pi_T) * C_fp))

    @staticmethod
    def from_json(data: dict) -> "BinaryGaussian":
        """
        Deserialize the classifier from a JSON-like dictionary.

        Args:
            data (dict): Dictionary containing the classifier data.

        Returns:
            BinaryGaussian: Deserialized classifier.
        """
        classifier = BinaryGaussian.__new__(BinaryGaussian)
        classifier.__classifier = data["classifier"]
        classifier.__S = data["S"]
        classifier.__pca_dimensions = data["pca_dimensions"]
        return classifier

    def to_json(self) -> dict:
        """
        Serialize the classifier to a JSON-like dictionary.

        Returns:
            dict: Dictionary containing the classifier data.
        """
        return {
            "classifier": self.__classifier,
            "pca_dimensions": self.__pca_dimensions,
            "S": self.__S.tolist(),
        }
