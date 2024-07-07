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

import numpy as np
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from project.classifiers.gaussian_mixture_model import GaussianMixtureModel
from project.figures.plots import scatter_3d
from project.funcs.base import load_data, split_db_2to1
from project.funcs.dcf import dcf


def lab10(DATA: str):
    X, y = load_data(DATA)

    (X_train, y_train), (X_val, y_val) = split_db_2to1(X.T, y)

    PRIOR = 0.1

    num_components = [1, 2, 4, 8, 16]

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
                min_dcf = dcf(scores, y_val, PRIOR, 1.0, 1.0, "min", True)

                min_dcfs_with_combinations.append(
                    {
                        "minDCF": min_dcf,
                        "components_false": components_false,
                        "components_true": components_true,
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

    print(
        f"Best combination: {min_dcfs_with_combinations[np.argmin([c['minDCF'] for c in min_dcfs_with_combinations])]}"
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
                min_dcf = dcf(scores, y_val, PRIOR, 1.0, 1.0, "min", True)

                min_dcfs_with_combinations.append(
                    {
                        "minDCF": min_dcf,
                        "components_false": components_false,
                        "components_true": components_true,
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

    print(
        f"Best combination: {min_dcfs_with_combinations[np.argmin([c['minDCF'] for c in min_dcfs_with_combinations])]}"
    )
