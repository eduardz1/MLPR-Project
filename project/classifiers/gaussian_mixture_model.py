import json
from dataclasses import dataclass, field
from io import TextIOWrapper
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.special as sp

from project.classifiers.classifier import Classifier
from project.funcs.base import cov, vcol, vrow
from project.funcs.log_pdf import log_pdf_gaussian, log_pdf_gmm


class GaussianMixtureModel(Classifier):
    """
    Creates a Gaussian Mixture Model

    Attributes:
        gmms (list[SingleGMM]): List of SingleGMM objects.
        llr (npt.NDArray): Log likelihood ratio of the classifier.

        _S (npt.NDArray): Scores of the classifier.
        _fitted (bool): Whether the classifier has been fitted or not.
    """

    def __init__(self) -> None:
        self.gmms = [SingleGMM() for _ in [0, 1]]

    def scores(self, X):
        self._S = np.array([gmm.scores(X) for gmm in self.gmms])

        return self._S

    @property
    def llr(self):
        """
        Log likelihood ratio of the classifier.
        llr(xₜ) = log(GMM(xₜ|M₁,S₁,w₁) / GMM(xₜ|M₀,S₀,w₀))
        """
        if not hasattr(self, "_S"):
            raise ValueError("Scores have not been computed yet.")

        return self._S[1] - self._S[0]

    def fit(
        self,
        X: npt.NDArray,
        y: npt.ArrayLike,
        *,
        num_components: list[int] | None = None,
        **kwargs,
    ) -> "GaussianMixtureModel":
        """
        Fits the Gaussian Mixture Model by training each SingleGMM object

        Args:
            X (npt.NDArray): Training data.
            y (npt.ArrayLike): Training labels.
            num_components (list[int] | None, optional): The number of components
                to use in the GMM for each class, default order is [false, true]
                (0, 1, ...). If None, the number of components is 1 for each class.
                Defaults to None.
            apply_lbg (bool, optional): Whether to apply the Linde-Buzo-Gray
                algorithm to initialize the GMM. Defaults to False.
            cov_type (
                Literal[full] | Literal[diagonal] | Literal[tied], optional
                ): The type of covariance matrix to use. Defaults to "full".
            psi_eig (float | None, optional): The minimum eigenvalue for the
                covariance matrix. Defaults to None.
            eps_ll_avg (float, optional): The minimum difference between the
                log-likelihoods of two consecutive iterations. Defaults to 1e-6.

        Returns:
            GaussianMixtureModel: The trained Gaussian Mixture Model.
        """
        if num_components is None:
            for i, gmm in enumerate(self.gmms):
                gmm.fit(X[:, y == i], **kwargs)
        else:
            # Train each GMM with the specified number of components
            for i, (gmm, num_component) in enumerate(zip(self.gmms, num_components)):
                gmm.fit(X[:, y == i], num_components=num_component, **kwargs)

        self._fitted = True

        return self

    def to_json(self, fp=None):
        if not self._fitted:
            raise ValueError("Classifier has not been fitted yet.")

        data = {"gmms": {i: gmm.to_json() for i, gmm in enumerate(self.gmms)}}

        if fp is None:
            return data

        json.dump(data, fp)

    @staticmethod
    def from_json(data):
        decoded = (
            json.load(data) if isinstance(data, TextIOWrapper) else json.loads(data)
        )

        gmms = []
        for _, gmm in decoded["gmms"].items():
            gmms.append(SingleGMM.from_json(gmm))

        gmm = GaussianMixtureModel.__new__(GaussianMixtureModel)
        gmm.gmms = gmms

        return gmm


@dataclass
class SingleGMM(Classifier):
    """
    Creates a Single Gaussian Mixture Model

    Attributes:
        params (list[tuple[npt.NDArray, npt.NDArray, npt.NDArray]]): List of
            tuples containing the weights, means, and covariances of the GMM.

        _type (str): The type of covariance matrix.
        _fitted (bool): Whether the classifier has been fitted or not.
        _S (npt.NDArray): Scores of the classifier.
    """

    params: list[tuple[npt.NDArray, npt.NDArray, npt.NDArray]] = field(
        default_factory=list
    )

    def scores(self, X):
        return log_pdf_gmm(X, self.params)

    @property
    def llr(self):
        raise NotImplementedError

    def fit(
        self,
        X: npt.NDArray,
        *,
        apply_lbg: bool = False,
        num_components: int = 1,
        cov_type: Literal["full"] | Literal["diagonal"] | Literal["tied"] = "full",
        eps_ll_avg: float = 1e-6,
        psi_eig: float | None = None,
    ) -> "SingleGMM":
        """
        Fits the Gaussian Mixture Model using the Expectation-Maximization algorithm

        Args:
            X (npt.NDArray): The training data.
            apply_lbg (bool, optional): Whether to apply the Linde-Buzo-Gray
                algorithm to initialize the GMM. Defaults to False.
            num_components (int, optional): The number of components to use in
                the GMM when applying the LBG algorithm. Defaults to 1.
            cov_type (
                Literal[full] | Literal[diagonal] | Literal[tied], optional
                ): The type of covariance matrix to use. Defaults to "full".
            psi_eig (float | None, optional): The minimum eigenvalue for the
                covariance matrix. Defaults to None.
            eps_ll_avg (float, optional): The minimum difference between the
                log-likelihoods of two consecutive iterations. Defaults to 1e-6.

        Returns:
            SingleGMM: The trained Single Gaussian Mixture Model.
        """
        self._type = cov_type

        if apply_lbg:
            mu = vcol(np.mean(X, axis=1))
            C = cov(X)

            if cov_type == "diagonal":
                C = C * np.eye(X.shape[0])

            self.params = [
                (
                    np.array(1.0),
                    mu,
                    C if psi_eig is None else self.smooth_covariance_matrix(C, psi_eig),
                )
            ]

            while len(self.params) < num_components:
                self.__lbg_split()
                self.__train(X, cov_type, eps_ll_avg, psi_eig)
        else:
            self.__train(X, cov_type, eps_ll_avg, psi_eig)

        self._fitted = True

        return self

    def __train(
        self, X: npt.NDArray, cov_type: str, eps_ll_avg: float, psi_eig: float | None
    ):
        """
        Trains the Gaussian Mixture Model using the Expectation-Maximization algorithm
        """

        while True:
            old_ll = np.mean(log_pdf_gmm(X, self.params))
            self.__em_it(X, cov_type, psi_eig)
            new_ll = np.mean(log_pdf_gmm(X, self.params))

            if new_ll - old_ll < eps_ll_avg:
                break

    def __em_it(self, X, cov_type, psi_eig):
        """
        Applies one iteration of the Expectation-Maximization algorithm
        """

        # E-step
        S = [log_pdf_gaussian(X, mu, C) + np.log(w) for w, mu, C in self.params]

        S = np.vstack(S)

        log_densities = sp.logsumexp(S, axis=0)

        # Posterior probabilities for all clusters, each row corresponds to a
        # Gaussian component, each column corresponds to a sample
        responsibilities = np.exp(S - log_densities)

        # M-step
        epsilon = 1e-6

        new_params: list[tuple] = []

        for resp in responsibilities:
            Z = resp.sum() + epsilon  # Prevent division by zero
            F = vcol((vrow(resp) * X).sum(1))
            S = (vrow(resp) * X) @ X.T

            # Update the parameters
            mu = F / Z
            C = S / Z - mu @ mu.T
            w = Z / X.shape[1]

            if cov_type == "diagonal":
                C = C * np.eye(X.shape[0])

            new_params.append((w, mu, C))

        if cov_type == "tied":
            C_tied = sum(w * C for w, _, C in new_params)
            new_params = [(w, mu, C_tied) for w, mu, _ in new_params]

        if psi_eig is not None:
            new_params = [
                (w, mu, self.smooth_covariance_matrix(C, psi_eig))
                for w, mu, C in new_params
            ]

        self.params = new_params

    def __lbg_split(self, alpha=0.1):
        """
        Splits the components of the GMM using the Linde-Buzo-Gray algorithm

        Args:
            alpha (float, optional): The scaling factor for the displacement.
                Defaults to 0.1.
        """

        new_params = []
        for w, mu, C in self.params:
            U, s, _ = np.linalg.svd(C)
            displacement = U[:, :1] * s[0] ** 0.5 * alpha
            new_params.append((0.5 * w, mu - displacement, C))
            new_params.append((0.5 * w, mu + displacement, C))

        self.params = new_params

    @staticmethod
    def smooth_covariance_matrix(C: npt.NDArray, psi: float):
        """
        Smooths the covariance matrix by setting the eigenvalues below a
        certain threshold to that threshold value

        Args:
            C (npt.NDArray): The covariance matrix
            psi (float): The minimum eigenvalue

        Returns:
            npt.NDArray: The smoothed covariance matrix
        """

        U, s, _ = np.linalg.svd(C)
        s[s < psi] = psi
        return U @ (vcol(s) * U.T)

    def to_json(self) -> dict:
        """
        Serializes the Single Gaussian Mixture Model to a JSON like dictionary

        Returns:
            dict: The JSON like dictionary
        """
        return {
            "type": self._type,
            "params": [
                {
                    "w": w.tolist(),
                    "mu": mu.tolist(),
                    "C": C.tolist() if self._type == "full" else np.diag(C).tolist(),
                }
                for w, mu, C in self.params
            ],
        }

    @staticmethod
    def from_json(data: dict) -> "SingleGMM":
        """
        Deserializes the Single Gaussian Mixture Model from a JSON like dictionary

        Args:
            data (dict): The JSON like dictionary to deserialize

        Returns:
            SingleGMM: The deserialized Single Gaussian Mixture Model
        """
        gmm = SingleGMM.__new__(SingleGMM)
        gmm._type = data["type"]
        gmm.params = [
            (
                np.array(d["w"]),
                np.array(d["mu"]),
                np.array(d["C"]) if gmm._type == "full" else np.diag(np.array(d["C"])),
            )
            for d in data["params"]
        ]
        return gmm
