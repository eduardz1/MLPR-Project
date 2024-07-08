from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.special as sp

from project.funcs.base import cov, vcol, vrow
from project.funcs.log_pdf import log_pdf_gaussian, log_pdf_gmm


class GaussianMixtureModel:
    def __init__(
        self, X_train: npt.NDArray, y_train: npt.NDArray, X_val: npt.NDArray
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val

        classes = np.unique(y_train)

        self.gmms = [SingleGMM(X_train[:, y_train == c]) for c in classes]

    @property
    def log_likelihood_ratio(self) -> npt.NDArray:
        """
        Log likelihood ratio of the classifier. llr(xₜ) = log(GMM(xₜ|M₁,S₁,w₁) / GMM(xₜ|M₀,S₀,w₀))
        """
        return np.array(log_pdf_gmm(self.X_val, self.gmms[1].params)) - np.array(
            log_pdf_gmm(self.X_val, self.gmms[0].params)
        )

    def train(self, num_components: list[int] | None = None, **kwargs) -> None:
        if num_components is None:
            for gmm in self.gmms:
                gmm.train(**kwargs)
        else:
            # Train each GMM with the specified number of components
            for gmm, num_component in zip(self.gmms, num_components):
                gmm.train(num_components=num_component, **kwargs)

    def to_json(self) -> dict:
        return {
            "gmms": [gmm.to_json() for gmm in self.gmms],
        }

    @staticmethod
    def from_json(data: dict) -> "GaussianMixtureModel":
        gmms = []
        for gmm in data["gmms"]:
            gmms.append(SingleGMM.from_json(gmm))

        gmm = GaussianMixtureModel.__new__(GaussianMixtureModel)
        gmm.gmms = gmms

        return gmm


@dataclass
class SingleGMM:
    X: npt.NDArray
    params: list[tuple[npt.NDArray, npt.NDArray, npt.NDArray]] = field(
        default_factory=list
    )

    def train(
        self,
        *,
        apply_lbg: bool = False,
        num_components: int = 1,
        cov_type: Literal["full"] | Literal["diagonal"] | Literal["tied"] = "full",
        eps_ll_avg: float = 1e-6,
        psi_eig: float | None = None,
    ):
        """
        Trains the Gaussian Mixture Model using the Expectation-Maximization algorithm

        Args:
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
        """

        if apply_lbg:
            mu = vcol(np.mean(self.X, axis=1))
            C = cov(self.X)

            if cov_type == "diagonal":
                C = C * np.eye(self.X.shape[0])

            self.params = [
                (
                    np.array(1.0),
                    mu,
                    C if psi_eig is None else self.smooth_covariance_matrix(C, psi_eig),
                )
            ]

            while len(self.params) < num_components:
                self.__lgb_split()
                self.__train(cov_type, eps_ll_avg, psi_eig)
        else:
            self.__train(cov_type, eps_ll_avg, psi_eig)

    def __train(self, cov_type, eps_ll_avg, psi_eig):
        """
        Trains the Gaussian Mixture Model using the Expectation-Maximization algorithm
        """

        while True:
            old_ll = np.mean(log_pdf_gmm(self.X, self.params))
            self.__em_it(cov_type, psi_eig)
            new_ll = np.mean(log_pdf_gmm(self.X, self.params))

            if new_ll - old_ll < eps_ll_avg:
                break

    def __em_it(self, cov_type, psi_eig):
        """
        Applies one iteration of the Expectation-Maximization algorithm
        """

        # E-step
        S = [log_pdf_gaussian(self.X, mu, C) + np.log(w) for w, mu, C in self.params]

        S = np.vstack(S)

        log_densities = sp.logsumexp(S, axis=0)

        # Posterior probabilities for all clusters, each row corresponds to a
        # Gaussian component, each column corresponds to a sample
        responsibilities = np.exp(S - log_densities)

        epsilon = 1e-6

        # M-step
        new_params = []
        for resp in responsibilities:
            Z = resp.sum() + epsilon  # Prevent division by zero
            F = vcol((vrow(resp) * self.X).sum(1))
            S = (vrow(resp) * self.X) @ self.X.T

            # Update the parameters
            mu = F / Z
            C = S / Z - mu @ mu.T
            w = Z / self.X.shape[1]

            if cov_type.lower() == "diagonal":
                C = C * np.eye(self.X.shape[0])

            new_params.append((w, mu, C))

        if cov_type.lower() == "tied":
            C_tied = sum(w * C for w, _, C in new_params)

            for i, (w, mu, _) in enumerate(new_params):
                new_params[i] = (w, mu, C_tied)

        if psi_eig is not None:
            for i, (w, mu, C) in enumerate(new_params):
                new_params[i] = (w, mu, self.smooth_covariance_matrix(C, psi_eig))

        self.params = new_params

    def __lgb_split(self, alpha=0.1):
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
        Smoothes the covariance matrix by setting the eigenvalues below a
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
        return {
            "params": [
                (w.tolist(), mu.tolist(), C.tolist()) for w, mu, C in self.params
            ]
        }

    @staticmethod
    def from_json(data: dict) -> "SingleGMM":
        gmm = SingleGMM.__new__(SingleGMM)
        gmm.params = data["params"]
        return gmm
