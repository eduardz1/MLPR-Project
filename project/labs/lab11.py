"""
# Calibration and Fusion

Consider the different classifiers that you trained in previous laboratories.
For each of the main methods (GMM, logistic regression, SVM — see Laboratory 10)
compute a calibration transformation for the scores of the best-performing
classifier you selected earlier. The calibration model should be trained using
the validation set that you employed in previous laboratories (i.e., the
validation split that you used to measure the systems performance). Apply a
K-fold approach to compute and evaluate the calibration transformation. You can
test different priors for training the logistic regression model, and evaluate
the performance of the calibration transformation in terms of actual DCF for the
target application (i.e., the training prior may be different than the target
application prior, but evaluation should be done for the target application).
For each model, select the best performing calibration transformation (i.e. the
one providing lowest actual DCF in the K-fold cross validation procedure for the
target application). Compute also the minimum DCF, and compare it to the actual
DCF, of the calibrated scores for the different systems. What do you observe?
Has calibration improved for the target application? What about different
applications (Bayes error plots)?

Compute a score-level fusion of the best-performing models. Again, you can try
different priors for training logistic regression, but you should select the
best model in terms of actual DCF computed for the target application. Compute
also the minimum DCF of the resulting model. How is the fusion performing? Is it
improving actual DCF with respect to single systems? Are the fused scores well
calibrated?

Choose the final model that will be used as “delivered” system, i.e. the final
system that will be used for application data. Justify your choice.
"""

import json
from typing import Literal

import numpy as np
import numpy.typing as npt
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from project.classifiers.logistic_regression import LogisticRegression
from project.figures.plots import plot
from project.funcs.base import load_data, split_db_2to1, vrow
from project.funcs.dcf import bayes_error_plot


# Extract i-th fold from a 1-D numpy array (as for the single fold case, we do
# not need to shuffle scores in this case, but it may be necessary if samples
# are sorted in peculiar ways to ensure that validation and calibration sets are
# independent and with similar characteristics)
def extract_train_val_folds_from_ary(K, X, idx):
    return np.ascontiguousarray(
        np.hstack([X[jdx::K] for jdx in range(K) if jdx != idx])
    ), np.ascontiguousarray(X[idx::K])


def lab11(DATA: str):
    X, y = load_data(DATA)

    (_, _), (_, y_val) = split_db_2to1(X.T, y)

    priors = np.linspace(0.1, 0.9, 9)

    K = 5

    def calibrate_and_plot(
        model: Literal["log_reg", "svm", "gmm"],
        priors: npt.NDArray,
        y_val: npt.NDArray,
        K: int,
    ):
        config = json.load(open(f"configs/best_{model}_config.json"))

        scores = np.array(config["scores"])

        log_odds, act_dcf, min_dcf = bayes_error_plot(scores, y_val)

        best_dcf = {
            "actDCF (pre calibration)": act_dcf,
            "minDCF": min_dcf,
            "actDCF (post calibration)": None,
        }
        best_prior = None

        with Progress(
            SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(f"Calibrating {model}...", total=len(priors) * K)

            for prior in priors:
                calibrated_scores = []
                calibrated_labels = []

                for i in range(K):
                    SCAL, SVAL = extract_train_val_folds_from_ary(K, scores, i)
                    LCAL, LVAL = extract_train_val_folds_from_ary(K, y_val, i)

                    log_reg = LogisticRegression(vrow(SCAL), LCAL, vrow(SVAL), LVAL)

                    log_reg.train(0, prior, prior_weighted=True)

                    calibrated_scores.append(log_reg.log_likelihood_ratio)
                    calibrated_labels.append(LVAL)

                    progress.update(task, advance=1)

                calibrated_scores = np.hstack(calibrated_scores)
                calibrated_labels = np.hstack(calibrated_labels)

                _, act_dcf_cal, _ = bayes_error_plot(
                    calibrated_scores, calibrated_labels
                )

                if (
                    best_dcf["actDCF (post calibration)"] is None
                    or act_dcf_cal < best_dcf["actDCF (post calibration)"]
                ):
                    best_dcf["actDCF (post calibration)"] = act_dcf_cal
                    best_prior = prior

        plot(
            best_dcf,
            log_odds,
            file_name=f"calibration_{model}_pi={best_prior}",
            xlabel="Effective Prior Log Odds",
            ylabel="DCF",
            marker="",
        )

    calibrate_and_plot("log_reg", priors, y_val, K)

    calibrate_and_plot("svm", priors, y_val, K)

    calibrate_and_plot("gmm", priors, y_val, K)

    # Fusion of the 3 models

    best_log_reg_config = json.load(open("configs/best_log_reg_config.json"))
    best_svm_config = json.load(open("configs/best_svm_config.json"))
    best_gmm_config = json.load(open("configs/best_gmm_config.json"))

    best_log_reg_scores = np.array(best_log_reg_config["scores"])
    best_svm_scores = np.array(best_svm_config["scores"])
    best_gmm_scores = np.array(best_gmm_config["scores"])

    with Progress(
        SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("Fusing models...", total=len(priors) * K)

        best_dcf = {
            "minDCF (fused scores)": [],
            "actDCF (fused scores)": [],
        }
        best_prior = None

        for prior in priors:
            fused_scores = []
            fused_labels = []

            for i in range(K):
                SCAL_L, SVAL_L = extract_train_val_folds_from_ary(
                    K, best_log_reg_scores, i
                )
                SCAL_S, SVAL_S = extract_train_val_folds_from_ary(K, best_svm_scores, i)
                SCAL_G, SVAL_G = extract_train_val_folds_from_ary(K, best_gmm_scores, i)

                LCAL, LVAL = extract_train_val_folds_from_ary(K, y_val, i)

                SCAL = np.vstack([SCAL_L, SCAL_S, SCAL_G])
                SVAL = np.vstack([SVAL_L, SVAL_S, SVAL_G])

                log_reg = LogisticRegression(SCAL, LCAL, SVAL, LVAL)

                log_reg.train(0, prior, prior_weighted=True)

                fused_scores.append(log_reg.log_likelihood_ratio)
                fused_labels.append(LVAL)

                progress.update(task, advance=1)

            fused_scores = np.hstack(fused_scores)
            fused_labels = np.hstack(fused_labels)

            log_odds, act_dcf_fus, min_dcf_fus = bayes_error_plot(
                fused_scores, fused_labels
            )

            if (
                not best_dcf["actDCF (fused scores)"]
                or act_dcf_fus < best_dcf["actDCF (fused scores)"]
            ):
                best_dcf["actDCF (fused scores)"] = act_dcf_fus
                best_dcf["minDCF (fused scores)"] = min_dcf_fus
                best_prior = prior

        plot(
            best_dcf,
            log_odds,
            file_name=f"fusion_pi={best_prior}",
            xlabel="Effective Prior Log Odds",
            ylabel="DCF",
            marker="",
        )
