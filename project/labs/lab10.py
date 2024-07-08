"""
# Gaussian Mixture Models

In this section we apply the GMM models to classification of the project data.

For each of the two classes, we need to decide the number of Gaussian components
(hyperparameter of the model). Train full covariance models with different
number of components for each class (suggestion: to avoid excessive training
time you can restrict yourself to models with up to 32 components). Evaluate the
performance on the validation set to perform model selection (again, you can use
the minimum DCF of the different models for the target application). Repeat the
analysis for diagonal models. What do you observe? Are there combinations which
work better? Are the results in line with your expectation, given the
characteristics that you observed in the dataset? Are there results that are
surprising? (Optional) Can you find an explanation for these surprising results?

We have analyzed all the classifiers of the course. For each of the main methods
(GMM, logistic regression, SVM — we ignore MVG since its results should be
significantly worse than those of the other models, but feel free to test it as
well) select the best performing candidate. Compare the models in terms of
minimum and actual DCF. Which is the most promising method for the given
application?

Now consider possible alternative applications. Perform a qualitative analysis
of the performance of the three approaches for different applications (keep the
models that you selected in the previous step). You can employ a Bayes error
plot and visualize, for each model, actual and minimum DCF over a wide range of
operating points (e.g. log-odds ranging from −4 to +4). What do you observe? In
terms of minimum DCF, are the results consistent, preserving the relative
ranking of the systems? What about actual DCF? Are there models that are well
calibrated for most of the operating point range? Are there models that show
significant miscalibration? Are there models that are harmful for some
applications? We will see how to deal with these issue in the last laboratory.
"""

import json

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from project.classifiers.gaussian_mixture_model import GaussianMixtureModel
from project.figures.plots import plot, scatter_3d
from project.figures.rich import table
from project.funcs.base import load_data, split_db_2to1
from project.funcs.dcf import bayes_error_plot, dcf


def lab10(DATA: str):
    console = Console()

    X, y = load_data(DATA)

    (X_train, y_train), (X_val, y_val) = split_db_2to1(X.T, y)

    PRIOR = 0.1

    num_components = [1, 2, 4, 8, 16]

    best_gmm_config = {
        "cov_type": "full",
        "min_dcf": np.inf,
        "act_dcf": np.inf,
        "components_false": 0,
        "components_true": 0,
        "scores": None,
        "model": None,
    }

    # training with "full" GMM

    min_dcfs_with_combinations = []

    with Progress(
        SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()
    ) as progress:
        task = progress.add_task(
            "Training full GMM models...", total=len(num_components) ** 2
        )

        for components_false in num_components:
            for components_true in num_components:
                gmm = GaussianMixtureModel(X_train, y_train, X_val)
                gmm.train(
                    apply_lbg=True,
                    num_components=[components_false, components_true],
                    cov_type="full",
                )
                scores = gmm.log_likelihood_ratio
                min_dcf = dcf(scores, y_val, PRIOR, 1.0, 1.0, "min")

                min_dcfs_with_combinations.append(
                    {
                        "minDCF": min_dcf,
                        "components_false": components_false,
                        "components_true": components_true,
                    }
                )

                if min_dcf < best_gmm_config["min_dcf"]:
                    best_gmm_config.update(
                        {
                            "cov_type": "full",
                            "min_dcf": min_dcf,
                            "act_dcf": dcf(scores, y_val, PRIOR, 1.0, 1.0, "optimal"),
                            "components_false": components_false,
                            "components_true": components_true,
                            "scores": scores.tolist(),
                            "model": gmm.to_json(),
                        }
                    )

                progress.update(task, advance=1)

    scatter_3d(
        [c["components_false"] for c in min_dcfs_with_combinations],
        [c["components_true"] for c in min_dcfs_with_combinations],
        [c["minDCF"] for c in min_dcfs_with_combinations],
        file_name="gmm/full",
        xlabel="Number of components (False)",
        ylabel="Number of components (True)",
        zlabel="minDCF",
    )

    # training with "diagonal" GMM

    min_dcfs_with_combinations = []

    with Progress(
        SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()
    ) as progress:
        task = progress.add_task(
            "Training diagonal GMM models...", total=len(num_components) ** 2
        )

        for components_false in num_components:
            for components_true in num_components:
                gmm = GaussianMixtureModel(X_train, y_train, X_val)
                gmm.train(
                    apply_lbg=True,
                    num_components=[components_false, components_true],
                    cov_type="diagonal",
                )
                scores = gmm.log_likelihood_ratio
                min_dcf = dcf(scores, y_val, PRIOR, 1.0, 1.0, "min")

                min_dcfs_with_combinations.append(
                    {
                        "minDCF": min_dcf,
                        "components_false": components_false,
                        "components_true": components_true,
                    }
                )

                if min_dcf < best_gmm_config["min_dcf"]:
                    best_gmm_config.update(
                        {
                            "cov_type": "diagonal",
                            "min_dcf": min_dcf,
                            "act_dcf": dcf(scores, y_val, PRIOR, 1.0, 1.0, "optimal"),
                            "components_false": components_false,
                            "components_true": components_true,
                            "scores": scores.tolist(),
                            "model": gmm.to_json(),
                        }
                    )

                progress.update(task, advance=1)

    scatter_3d(
        [c["components_false"] for c in min_dcfs_with_combinations],
        [c["components_true"] for c in min_dcfs_with_combinations],
        [c["minDCF"] for c in min_dcfs_with_combinations],
        file_name="gmm/diagonal",
        xlabel="Number of components (False)",
        ylabel="Number of components (True)",
        zlabel="minDCF",
    )

    with open("configs/best_gmm_config.json", "w") as f:
        json.dump(best_gmm_config, f)

    scores = best_gmm_config.pop("scores")
    model = best_gmm_config.pop("model")
    table(console, "Best GMM configuration", best_gmm_config)
    best_gmm_config["scores"] = scores
    best_gmm_config["model"] = model

    # Analyze the best combinations of SVM, LogReg and GMM

    best_log_reg_config = json.load(open("configs/best_log_reg_config.json"))
    best_svm_config = json.load(open("configs/best_svm_config.json"))

    table(
        console,
        "Models comparison",
        {
            "DCF": ["min", "act"],
            "GMM": [best_gmm_config["min_dcf"], best_gmm_config["act_dcf"]],
            "LogReg": [best_log_reg_config["min_dcf"], best_log_reg_config["act_dcf"]],
            "SVM": [best_svm_config["min_dcf"], best_svm_config["act_dcf"]],
        },
    )

    # Qualitative analysis on different applications for GMM

    log_odds, act_dcf_gmm, min_dcf_gmm = bayes_error_plot(
        np.array(best_gmm_config["scores"]), y_val
    )

    # Qualitative analysis on different applications for LogReg

    log_odds, act_dcf_log_reg, min_dcf_log_reg = bayes_error_plot(
        np.array(best_log_reg_config["scores"]), y_val
    )

    # Qualitative analysis on different applications for SVM

    log_odds, act_dcf_svm, min_dcf_svm = bayes_error_plot(
        np.array(best_svm_config["scores"]), y_val
    )

    plot(
        {
            "GMM": act_dcf_gmm,
            "LogReg": act_dcf_log_reg,
            "SVM": act_dcf_svm,
        },
        log_odds,
        file_name="models_act_dcf",
        xlabel="Effective prior log odds",
        ylabel="DCF",
        marker="",
    )

    plot(
        {
            "GMM": min_dcf_gmm,
            "LogReg": min_dcf_log_reg,
            "SVM": min_dcf_svm,
        },
        log_odds,
        file_name="models_min_dcf",
        xlabel="Effective prior log odds",
        ylabel="DCF",
        marker="",
    )
