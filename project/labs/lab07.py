"""
# Performance analysis of the MVG classifier

Analyze the performance of the MVG classifier and its variants for different
applications. Start considering five applications, given by (π1, Cf_n, Cf_p):

-   (0.5, 1.0, 1.0), i.e., uniform prior and costs
-   (0.9, 1.0, 1.0), i.e., the prior probability of a genuine sample is higher
    (in our application, most users are legit)
-   (0.1, 1.0, 1.0), i.e., the prior probability of a fake sample is higher
    (in our application, most users are impostors)
-   (0.5, 1.0, 9.0), i.e., the prior is uniform (same probability of a legit and
    fake sample), but the cost of accepting a fake image is larger (granting
    access to an impostor has a higher cost than labeling as impostor a legit
    user - we aim for strong security)
-   (0.5, 9.0, 1.0), i.e., the prior is uniform (same probability of a legit and
    fake sample), but the cost of rejecting a legit image is larger (granting
    access to an impostor has a lower cost than labeling a legit user as
    impostor - we aim for ease of use for legit users)

Represent the applications in terms of effective prior. What do you obtain?
Observe how the costs of mis-classifications are reflected in the prior:
stronger security (higher false positive cost) corresponds to an equivalent
lower prior probability of a legit user.

We now focus on the three applications, represented in terms of effective priors
(i.e., with costs of errors equal to 1) given by ˜π = 0.1, ˜π = 0.5 and
˜π = 0.9, respectively.

For each application, compute the optimal Bayes decisions for the validation set
for the MVG models and its variants, with and without PCA (try different values
of m). Compute DCF (actual) and minimum DCF for the different models. Compare
the models in terms of minimum DCF. Which models perform best? Are relative
performance results consistent for the different applications? Now consider also
actual DCFs. Are the models well calibrated (i.e., with a calibration loss in
the range of few percents of the minimum DCF value) for the given applications?
Are there models that are better calibrated than others for the considered
applications?

Consider now the PCA setup that gave the best results for the ˜π = 0.1
configuration (this will be our main application). Compute the Bayes error plots
for the MVG, Tied and Naive Bayes Gaussian classifiers. Compare the minimum DCF
of the three models for different applications, and, for each model, plot
minimum and actual DCF. Consider prior log odds in the range (−4, +4). What do
you observe? Are model rankings consistent across applications (minimum DCF)?
Are models well-calibrated over the considered range?
"""

import numpy as np
from rich.console import Console

from project.classifiers.binary_gaussian import BinaryGaussian
from project.figures.plots import plot
from project.figures.rich import table
from project.funcs.base import load_data, split_db_2to1
from project.funcs.dcf import bayes_error, dcf
from project.funcs.dcf import effective_prior as effective_prior


def lab07(DATA: str):
    console = Console()

    X, y = load_data(DATA)

    (X_train, y_train), (X_val, y_val) = split_db_2to1(X.T, y)

    cl = BinaryGaussian(X_train, y_train, X_val, y_val)

    applications = {
        "pi_T": [0.5, 0.9, 0.1, 0.5, 0.5],
        "C_fn": [1.0, 1.0, 1.0, 1.0, 9.0],
        "C_fp": [1.0, 1.0, 1.0, 9.0, 1.0],
    }

    effective_priors = []
    for i in range(5):
        pi_T = applications["pi_T"][i]
        C_fn = applications["C_fn"][i]
        C_fp = applications["C_fp"][i]

        e = effective_prior(pi_T, C_fn, C_fp)
        effective_priors.append(e)

        console.print(
            f"Application {i+1} ({{'pi_T': {pi_T}, 'C_fn': {C_fn}, 'C_fp': {C_fp}}}) - Effective prior: {e}"
        )

    effective_priors = np.unique(effective_priors)

    best_setups: dict[float, tuple[float, str, int | None, float]] = {
        k: (np.inf, "", None, np.inf) for k in effective_priors
    }

    best_pca_01_setups: dict[str, tuple[float, int, float]] = {
        "multivariate": (np.inf, 0, np.inf),
        "tied": (np.inf, 0, np.inf),
        "naive": (np.inf, 0, np.inf),
    }

    # Analyze the Multivariate Gaussian Classifier

    cl.fit(classifier="multivariate")

    act_dcfs = dcf(cl.llr, y_val, effective_priors, "optimal")
    min_dcfs = dcf(cl.llr, y_val, effective_priors, "min")

    for i, pi in enumerate(effective_priors):
        if min_dcfs[i] < best_setups[pi][0]:
            best_setups[pi] = (min_dcfs[i], "multivariate", None, act_dcfs[i])

    for m in range(1, 7):
        cl.fit(classifier="multivariate", pca_dimensions=m)

        act_dcfs = dcf(cl.llr, y_val, effective_priors, "optimal")
        min_dcfs = dcf(cl.llr, y_val, effective_priors, "min")

        for i, pi in enumerate(effective_priors):
            if min_dcfs[i] < best_setups[pi][0]:
                best_setups[pi] = (act_dcfs[i], "multivariate", m, act_dcfs[i])

            if pi == 0.1 and min_dcfs[i] < best_pca_01_setups["multivariate"][0]:
                best_pca_01_setups["multivariate"] = (min_dcfs[i], m, act_dcfs[i])

    # Analyze the Tied Gaussian Classifier

    cl.fit(classifier="tied")

    act_dcfs = dcf(cl.llr, y_val, effective_priors, "optimal")
    min_dcfs = dcf(cl.llr, y_val, effective_priors, "min")

    for i, pi in enumerate(effective_priors):
        if min_dcfs[i] < best_setups[pi][0]:
            best_setups[pi] = (min_dcfs[i], "tied", None, act_dcfs[i])

    for m in range(1, 7):
        cl.fit(classifier="tied", pca_dimensions=m)

        act_dcfs = dcf(cl.llr, y_val, effective_priors, "optimal")
        min_dcfs = dcf(cl.llr, y_val, effective_priors, "min")

        for i, pi in enumerate(effective_priors):
            if min_dcfs[i] < best_setups[pi][0]:
                best_setups[pi] = (act_dcfs[i], "tied", m, act_dcfs[i])

            if pi == 0.1 and min_dcfs[i] < best_pca_01_setups["tied"][0]:
                best_pca_01_setups["tied"] = (min_dcfs[i], m, act_dcfs[i])

    # Analyze the Naive Gaussian Classifier

    cl.fit(classifier="naive")

    act_dcfs = dcf(cl.llr, y_val, effective_priors, "optimal")
    min_dcfs = dcf(cl.llr, y_val, effective_priors, "min")

    for i, pi in enumerate(effective_priors):
        if min_dcfs[i] < best_setups[pi][0]:
            best_setups[pi] = (min_dcfs[i], "naive", None, act_dcfs[i])

    for m in range(1, 7):
        cl.fit(classifier="naive", pca_dimensions=m)

        act_dcfs = dcf(cl.llr, y_val, effective_priors, "optimal")
        min_dcfs = dcf(cl.llr, y_val, effective_priors, "min")

        for i, pi in enumerate(effective_priors):
            if min_dcfs[i] < best_setups[pi][0]:
                best_setups[pi] = (act_dcfs[i], "naive", m, act_dcfs[i])

            if pi == 0.1 and min_dcfs[i] < best_pca_01_setups["naive"][0]:
                best_pca_01_setups["naive"] = (min_dcfs[i], m, act_dcfs[i])

    table(
        console,
        "Best Setups",
        {
            "Effective Prior": [k for k in best_setups],
            "Model": [v[1] for v in best_setups.values()],
            "PCA Dimensions": [v[2] for v in best_setups.values()],
            "Minimum DCF": [v[0] for v in best_setups.values()],
            "Actual DCF": [v[3] for v in best_setups.values()],
        },
    )

    table(
        console,
        "Best PCA Setups for 0.1 Effective Prior",
        {
            "Model": [k for k in best_pca_01_setups],
            "PCA Dimensions": [v[1] for v in best_pca_01_setups.values()],
            "Minimum DCF": [v[0] for v in best_pca_01_setups.values()],
            "Actual DCF": [v[2] for v in best_pca_01_setups.values()],
        },
    )

    for model in ["multivariate", "tied", "naive"]:
        cl.fit(classifier=model, pca_dimensions=best_pca_01_setups[model][1])  # type: ignore

        log_odds, act_dcf, min_dcf = bayes_error(cl.llr, y_val)

        plot(
            {
                "DCF": act_dcf.tolist(),
                "Min DCF": min_dcf.tolist(),
            },
            log_odds,
            colors=["purple", "purple"],
            linestyles=["solid", "dashed"],
            file_name=f"{model}_prior_log_odds",
            xlabel="Effective prior log odds",
            ylabel="DCF",
            marker="",
        )
