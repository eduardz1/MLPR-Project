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

from functools import partial
from pprint import pprint

import numpy as np
import scipy.optimize as opt

from project.figures.plots import plot
from project.funcs.base import load_data, split_db_2to1
from project.funcs.dcf import dcf
from project.funcs.logreg_obj import logreg_obj


def quadratic_feature_expansion(X):
    X = X.T

    # Compute the outer product for each vector with itself using einsum
    outer_products = np.einsum("ij,ik->ijk", X, X)

    flattened_outer_products = outer_products.reshape(X.shape[0], -1)
    X_panded = np.concatenate([flattened_outer_products, X], axis=1)

    return X_panded.T


def compute_logistic_regression(
    X_train, y_train, X_val, y_val, lambdas, prior, prior_weighted=False
):
    applications = {
        "λ": [],
        "J(w*,b*)": [],
        "Error rate": [],
        "minDCF": [],
        "actDCF": [],
    }

    for l in lambdas:
        logReg = partial(
            logreg_obj,
            approx_grad=True,
            DTR=X_train,
            LTR=y_train,
            l=l,
            prior=prior if prior_weighted else None,
        )

        x, f, _ = opt.fmin_l_bfgs_b(
            logReg,
            np.zeros(X_train.shape[0] + 1),
            approx_grad=True,
        )

        w, b = x[:-1], x[-1]

        S = w @ X_val + b
        LP = S > 0
        error_rate = np.mean(LP != y_val)

        if prior_weighted:
            S_llr = S.ravel() - np.log(prior / (1 - prior))
        else:
            pi_emp = np.mean(y_train)  # Fractions of samples of class 1
            S_llr = S.ravel() - np.log(pi_emp / (1 - pi_emp))

        min_dcf = dcf(S_llr, y_val, prior, 1, 1, "min", normalize=True)
        act_dcf = dcf(S_llr, y_val, prior, 1, 1, "optimal", normalize=True)

        applications["λ"].append(l)
        applications["J(w*,b*)"].append(f)
        applications["Error rate"].append(error_rate)
        applications["minDCF"].append(min_dcf)
        applications["actDCF"].append(act_dcf)

    return applications


def lab08(DATA: str):
    X, y = load_data(DATA)

    (X_train, y_train), (X_val, y_val) = split_db_2to1(X.T, y)
    X_train = np.ascontiguousarray(X_train)
    lambdas = np.logspace(-4, 2, 13)

    PRIOR = 0.1

    applications = compute_logistic_regression(
        X_train, y_train, X_val, y_val, lambdas, PRIOR
    )

    pprint(applications)
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
        X_train_s, y_train_s, X_val, y_val, lambdas, PRIOR
    )

    pprint(applications)

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
        X_train, y_train, X_val, y_val, lambdas, PRIOR, prior_weighted=True
    )

    pprint(applications)

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
    X_train_expanded = quadratic_feature_expansion(X_train)
    X_val_expanded = quadratic_feature_expansion(X_val)

    applications = compute_logistic_regression(
        X_train_expanded, y_train, X_val_expanded, y_val, lambdas, PRIOR
    )

    pprint(applications)

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

    # Centering the data and repeat the analysis
    X_train_centered = X_train - X_train.mean(axis=1, keepdims=True)
    X_val_centered = X_val - X_train.mean(axis=1, keepdims=True)

    applications = compute_logistic_regression(
        X_train_centered, y_train, X_val_centered, y_val, lambdas, PRIOR
    )

    pprint(applications)

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
