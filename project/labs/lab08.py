"""
# Performance analysis of the Binary Logistic Regression classifier

We analyze the binary logistic regression model on the project data. We start
considering the standard, non-weighted version of the model, without any
pre-processing.

Train the model using different values for λ. You can build logarthimic-spaced
values for λ using `numpy.logspace`. To obtain good coverage, you can use
`numpy.logspace(-4, 2, 13)` (check the documentation). Train the model with each
value of λ, score the validation samples and compute the corresponding actual
DCF and minimum DCF for the primary application πT = 0.1. To compute actual DCF
remember to remove the log-odds of the training set empirical prior. Plot the
two metrics as a function of λ (suggestion: use a logartihmic scale for the
x-axis of the plot - to change the scale of the x-axis you can use
`matplotlib.pyplot.xscale(’log’, base=10))`. What do you observe? Can you see
significant differences for the different values of λ? How does the
regularization coefficient affects the two metrics?

Since we have a large number of samples, regularization seems ineffective, and
actually degrades actual DCF since the regularized models tend to lose the
probabilistic interpretation of the scores. To better understand the role of
regularization, we analyze the results that we would obtain if we had fewer
training samples. Repeat the previous analysis, but keep only 1 out of 50 model
training samples, e.g. using data matrices `DTR[:, ::50]`, `LTR[::50]` (apply
the filter only on the model training samples, not on the validation samples,
i.e., after splitting the dataset in model training and validation sets). What
do you observe? Can you explain the results in this case? Remember that lower
values of the regularizer imply larger risk of overfitting, while higher values
of the regularizer reduce overfitting, but may lead to underfitting and to
scores that lose their probabilistic interpretation.

In the following we will again consider only the full dataset. Repeat the
analysis with the prior-weighted version of the model (remember that, in this
case, to transform the scores to LLRs you need to remove the log-odds of the
prior that you chose when training the model). Are there significant differences
for this task? Are there advantages using the prior-weighted model for our
application (remember that the prior-weighted model requires that we know the
target prior when we build the model)?

Repeat the analysis with the quadratic logistic regression model (again, full
dataset only). Expand the features, train and evaluate the models (you can focus
on the standard, non prior-weighted model only, as the results you would obtain
are similar for the two models), again considering different values for λ. What
do you observe? In this case is regularization effective? How does it affect the
two metrics?

The non-regularized model is invariant to affine transformations of the data.
However, once we introduce a regularization term affine transformations of the
data can lead to different results. Analyze the effects of centering
(optionally, you can also try different strategies, including Z-normalization
and whitening, as well as PCA) on the model results. You can restrict the
analysis to the linear model. Remember that you have to center both datasets
with respect to the model training dataset mean, i.e., you must not use the
validation data to estimate the pre-processing transformation. For this task,
you should observe only minor variations, as the original features were already
almost standardized.

As you should have observed, the best models in terms of minimum DCF are not
necessarily those that provide the best actual DCFs, i.e., they may present
significant mis-calibration. We will deal with score calibration at the end of
the course. For the moment, we focus on selecting the models that optimize the
minimum DCF on our validation set. Compare all models that you have trained up
to now, including Gaussian models, in terms of minDCF for the target application
πT = 0.1. Which model(s) achieve(s) the best results? What kind of separation
rules or distribution assumptions characterize this / these model(s)? How are
the results related to the characteristics of the dataset features?

Suggestion for upcoming laboratories: the last laboratories will cover score
calibration, and will require to evaluate the results of the models that you
tested on an held-out evaluation set. We suggest that you save the models, and
the corresponding validation scores as well, since these will be required for
score calibration (you can skip the models trained with the reduced dataset, as
they won’t be needed).
"""

import json

import numpy as np
from rich.console import Console

from project.classifiers.logistic_regression import LogisticRegression
from project.figures.plots import plot
from project.figures.rich import table
from project.funcs.base import load_data, quadratic_feature_expansion, split_db_2to1
from project.funcs.dcf import dcf


def compute_logistic_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    lambdas,
    prior,
    best_log_reg_config,
    prior_weighted=False,
    quadratic=False,
    centered=False,
):
    applications = {
        "λ": [],
        "J(w*,b*)": [],
        "Error rate": [],
        "minDCF": [],
        "actDCF": [],
    }

    if quadratic:
        X_train = quadratic_feature_expansion(X_train)
        X_val = quadratic_feature_expansion(X_val)

    if centered:
        X_train = X_train - X_train.mean(axis=1, keepdims=True)
        X_val = X_val - X_val.mean(axis=1, keepdims=True)

    cl = LogisticRegression(X_train, y_train, X_val, y_val)

    for l in lambdas:
        f = cl.train(l, prior, prior_weighted)

        min_dcf = dcf(cl.llr, y_val, prior, 1, 1, "min")
        act_dcf = dcf(cl.llr, y_val, prior, 1, 1, "optimal")

        applications["λ"].append(l)
        applications["J(w*,b*)"].append(f)
        applications["Error rate"].append(cl.error_rate)
        applications["minDCF"].append(min_dcf)
        applications["actDCF"].append(act_dcf)

        if min_dcf < best_log_reg_config["min_dcf"]:
            best_log_reg_config.update(
                {
                    "lambda": l,
                    "prior_weighted": prior_weighted,
                    "quadratic": quadratic,
                    "centered": centered,
                    "min_dcf": min_dcf,
                    "act_dcf": act_dcf,
                    "scores": cl.llr.tolist(),
                    "model": cl.to_json(),
                }
            )

    return applications


def lab08(DATA: str):
    console = Console()

    X, y = load_data(DATA)

    (X_train, y_train), (X_val, y_val) = split_db_2to1(X.T, y)
    X_train = np.ascontiguousarray(X_train)
    lambdas = np.logspace(-4, 2, 13)

    PRIOR = 0.1

    best_log_reg_config = {
        "lambda": 0,
        "prior_weighted": False,
        "quadratic": False,
        "centered": False,
        "min_dcf": np.inf,
        "act_dcf": np.inf,
        "scores": None,
        "model": None,
    }

    # Compute the standard logistic regression model

    applications = compute_logistic_regression(
        X_train, y_train, X_val, y_val, lambdas, PRIOR, best_log_reg_config
    )

    table(console, "Logistic regression", applications)

    plot(
        {
            "minDCF": applications["minDCF"],
            "actDCF": applications["actDCF"],
        },
        applications["λ"],
        file_name="lambda_vs_dcf",
        xscale="log",
        xlabel="λ",
        ylabel="DCF",
    )

    # Take only 1 in 50 samples
    X_train_s, y_train_s = X_train[:, ::50], y_train[::50]

    applications = compute_logistic_regression(
        X_train_s, y_train_s, X_val, y_val, lambdas, PRIOR, best_log_reg_config
    )

    table(console, "Logistic regression (1 in 50)", applications)

    plot(
        {
            "minDCF": applications["minDCF"],
            "actDCF": applications["actDCF"],
        },
        applications["λ"],
        file_name="lambda_vs_dcf_50",
        xscale="log",
        xlabel="λ",
        ylabel="DCF",
    )

    # Repeat with the prior-weighted version of the model
    applications = compute_logistic_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        lambdas,
        PRIOR,
        best_log_reg_config,
        prior_weighted=True,
    )

    table(console, "Prior-weighted logistic regression", applications)

    plot(
        {
            "minDCF": applications["minDCF"],
            "actDCF": applications["actDCF"],
        },
        applications["λ"],
        file_name="lambda_vs_dcf_prior",
        xscale="log",
        xlabel="λ",
        ylabel="DCF",
    )

    # Compute the quadratic logistic regression model

    applications = compute_logistic_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        lambdas,
        PRIOR,
        best_log_reg_config,
        quadratic=True,
    )

    table(console, "Quadratic logistic regression", applications)

    plot(
        {
            "minDCF": applications["minDCF"],
            "actDCF": applications["actDCF"],
        },
        applications["λ"],
        file_name="lambda_vs_dcf_quadratic",
        xscale="log",
        xlabel="λ",
        ylabel="DCF",
    )

    # Quadratic feature expanded and prior weighted
    applications = compute_logistic_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        lambdas,
        PRIOR,
        best_log_reg_config,
        prior_weighted=True,
        quadratic=True,
    )

    table(console, "Quadratic logistic regression (prior weighted)", applications)

    plot(
        {
            "minDCF": applications["minDCF"],
            "actDCF": applications["actDCF"],
        },
        applications["λ"],
        file_name="lambda_vs_dcf_quadratic_prior",
        xscale="log",
        xlabel="λ",
        ylabel="DCF",
    )

    # Centering the data and repeat the analysis

    applications = compute_logistic_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        lambdas,
        PRIOR,
        best_log_reg_config,
        centered=True,
    )

    table(console, "Centered logistic regression", applications)

    plot(
        {
            "minDCF": applications["minDCF"],
            "actDCF": applications["actDCF"],
        },
        applications["λ"],
        file_name="lambda_vs_dcf_centered",
        xscale="log",
        xlabel="λ",
        ylabel="DCF",
    )

    with open("configs/best_log_reg_config.json", "w") as f:
        json.dump(best_log_reg_config, f)

    best_log_reg_config.pop("scores")
    best_log_reg_config.pop("model")
    table(console, "Best logistic regression setup", best_log_reg_config)
