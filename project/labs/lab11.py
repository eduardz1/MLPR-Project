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

## Evaluation

We now evaluate the final delivered system, and perform further analysis to
understand whether our design choices were indeed good for our application.
The file `Project/evalData.txt` contains an evaluation dataset (with the same
format as the training dataset). Evaluate your chosen model on this dataset
(note: the evaluation dataset must **not** be used to estimate anything, we are
evaluating the models that we already trained).

    - Compute minimum and actual DCF, and Bayes error plots for the delivered
      system. What do you observe? Are scores well calibrated for the target
      application? And for other possible applications?

    - Consider the three best performing systems, and their fusion. Evaluate
      the corresponding actual DCF, and compare their actual DCF error plots.
      What do you observe? Was your final model choice effective? Would another
      model / fusion of models have been more effective?

    - Consider again the three best systems. Evaluate minimum and actual DCF
      for the target application, and analyze the corresponding Bayes error
      plots. What do you observe? Was the calibration strategy effective for
      the different approaches?

    - Now consider one of the three approaches (we should repeat this part of
      the analysis for all systems, but for the report you can consider only a
      single method). Analyze whether your training strategy was effective. For
      this, consider all models that you trained for the selected approach
      (e.g., if you chose the logistic regression method, the different
      hyper-parameter / pre-processing combinations of logistic regression
      models). Evaluate the minimum DCF of the considered systems on the
      evaluation, and compare it to the minimum DCF of the selected model (it
      would be better to analyze actual DCF, but this would require to
      re-calibrated all models, for brevity we skip this step). What do you
      observe? Was your chosen model optimal or close to optimal for the
      evaluation data? Were there different choices that would have led to
      better performance?
"""

import json
from typing import Literal

import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from project.classifiers.logistic_regression import LogisticRegression
from project.figures.plots import plot
from project.figures.rich import table
from project.funcs.base import load_data, split_db_2to1, vrow
from project.funcs.dcf import bayes_error_plot, dcf

TARGET_PRIOR = 0.1

BEST_LOG_REG_CONFIG = json.load(open("configs/best_log_reg_config.json"))
BEST_SVM_CONFIG = json.load(open("configs/best_svm_config.json"))
BEST_GMM_CONFIG = json.load(open("configs/best_gmm_config.json"))


# Extract i-th fold from a 1-D numpy array (as for the single fold case, we do
# not need to shuffle scores in this case, but it may be necessary if samples
# are sorted in peculiar ways to ensure that validation and calibration sets are
# independent and with similar characteristics)
def extract_fold(K, X, idx):
    return np.ascontiguousarray(
        np.hstack([X[jdx::K] for jdx in range(K) if jdx != idx])
    ), np.ascontiguousarray(X[idx::K])


def find_best_prior_and_plot_bayes_error(
    priors: npt.NDArray,
    y_val: npt.NDArray,
    K: int,
    model: Literal["log_reg", "svm", "gmm", "fusion"] = "fusion",
    fusion: bool = False,
):
    if not fusion:
        scores = np.array(eval(f"BEST_{model.upper()}_CONFIG")["scores"])
        min_dcf_target = dcf(scores, y_val, TARGET_PRIOR, "min").item()
        log_odds, act_dcf_applications, min_dcf_applications = bayes_error_plot(
            scores, y_val
        )

    best_prior = None
    best_act_dcf_target = np.inf
    best_calibrated_scores = np.array([])
    best_calibrated_labels = np.array([])

    applications = (
        {
            "minDCF (fused scores)": [],
            "actDCF (fused scores)": [],
        }
        if fusion
        else {
            "actDCF (pre calibration)": act_dcf_applications,
            "minDCF": min_dcf_applications,
            "actDCF (post calibration)": [],
        }
    )

    with Progress(
        SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()
    ) as progress:
        task = progress.add_task(
            "Fusing models..." if fusion else f"Calibrating {model}...",
            total=len(priors) * K,
        )

        for prior in priors:
            calibrated_scores = []
            calibrated_labels = []

            calibrated_scores, calibrated_labels = kfold_calibration(
                model,
                y_val,
                K,
                fusion,
                progress,
                task,
                prior,
                calibrated_scores,
                calibrated_labels,
            )

            act_dcf = dcf(
                calibrated_scores, calibrated_labels, TARGET_PRIOR, "optimal"
            ).item()

            if act_dcf < best_act_dcf_target:
                best_act_dcf_target = act_dcf
                best_prior = prior
                best_calibrated_scores = calibrated_scores
                best_calibrated_labels = calibrated_labels

        log_odds, act_dcf_applications, min_dcf_applications = bayes_error_plot(
            best_calibrated_scores, best_calibrated_labels
        )

        if fusion:
            applications["actDCF (fused scores)"] = act_dcf_applications.tolist()
            applications["minDCF (fused scores)"] = min_dcf_applications.tolist()
        else:
            applications["actDCF (post calibration)"] = act_dcf_applications.tolist()

        plot(
            applications,
            log_odds,
            colors=["b", "b"] if fusion else ["b", "b", "b"],
            linestyles=["dashed", "solid"] if fusion else ["dotted", "dashed", "solid"],
            file_name=f"calibration/{model}",
            xlabel="Effective Prior Log Odds",
            ylabel="DCF",
            marker="",
        )

        if fusion:
            return best_prior, best_act_dcf_target
        else:
            return best_prior, best_act_dcf_target, min_dcf_target


def kfold_calibration(
    model,
    y_val,
    K,
    fusion,
    progress,
    task,
    prior,
    calibrated_scores,
    calibrated_labels,
):
    if fusion:
        best_log_reg_scores = np.array(BEST_LOG_REG_CONFIG["scores"])
        best_svm_scores = np.array(BEST_SVM_CONFIG["scores"])
        best_gmm_scores = np.array(BEST_GMM_CONFIG["scores"])
    else:
        scores = np.array(eval(f"BEST_{model.upper()}_CONFIG")["scores"])

    for i in range(K):
        LCAL, LVAL = extract_fold(K, y_val, i)

        if fusion:
            SCAL_L, SVAL_L = extract_fold(K, best_log_reg_scores, i)
            SCAL_S, SVAL_S = extract_fold(K, best_svm_scores, i)
            SCAL_G, SVAL_G = extract_fold(K, best_gmm_scores, i)

            SCAL = np.vstack([SCAL_L, SCAL_S, SCAL_G])
            SVAL = np.vstack([SVAL_L, SVAL_S, SVAL_G])
        else:
            SCAL, SVAL = extract_fold(K, scores, i)

            SCAL = vrow(SCAL)
            SVAL = vrow(SVAL)

        log_reg = LogisticRegression(SCAL, LCAL, SVAL, LVAL)

        log_reg.train(0, prior, prior_weighted=True)

        calibrated_scores.append(log_reg.llr)
        calibrated_labels.append(LVAL)

        progress.update(task, advance=1)

    calibrated_scores = np.hstack(calibrated_scores)
    calibrated_labels = np.hstack(calibrated_labels)
    return calibrated_scores, calibrated_labels


def lab11(DATA: str):
    console = Console()

    X, y = load_data(DATA)

    (_, _), (_, y_val) = split_db_2to1(X.T, y)

    priors = np.linspace(0.1, 0.9, 9)

    K = 5

    models = [
        {
            "model": "log_reg",
            "prior": None,
            "actDCF": np.inf,
            "minDCF": np.inf,
        },
        {
            "model": "svm",
            "prior": None,
            "actDCF": np.inf,
            "minDCF": np.inf,
        },
        {
            "model": "gmm",
            "prior": None,
            "actDCF": np.inf,
            "minDCF": np.inf,
        },
        {
            "model": "fusion",
            "prior": None,
            "actDCF": np.inf,
        },
    ]

    log_reg_cal_prior, log_reg_act_dcf, log_reg_min_dcf = (  # type: ignore
        find_best_prior_and_plot_bayes_error(priors, y_val, K, "log_reg")
    )
    models[0]["actDCF"] = log_reg_act_dcf
    models[0]["minDCF"] = log_reg_min_dcf
    models[0]["prior"] = log_reg_cal_prior

    svm_cal_prior, svm_act_dcf, svm_min_dcf = find_best_prior_and_plot_bayes_error(  # type: ignore
        priors, y_val, K, "svm"
    )
    models[1]["actDCF"] = svm_act_dcf
    models[1]["minDCF"] = svm_min_dcf
    models[1]["prior"] = svm_cal_prior

    gmm_cal_prior, gmm_act_dcf, gmm_min_dcf = find_best_prior_and_plot_bayes_error(  # type: ignore
        priors, y_val, K, "gmm"
    )
    models[2]["actDCF"] = gmm_act_dcf
    models[2]["minDCF"] = gmm_min_dcf
    models[2]["prior"] = gmm_cal_prior

    fusion_cal_prior, fusion_act_dcf = find_best_prior_and_plot_bayes_error(  # type: ignore
        priors, y_val, K, fusion=True
    )
    models[3]["actDCF"] = fusion_act_dcf
    models[3]["prior"] = fusion_cal_prior

    table(
        console, "Best Model after Calibration", min(models, key=lambda x: x["actDCF"])
    )

    # (X_eval, y_eval) = load_data("data/evalData.txt")

    # log_reg = delivery_model["calibration_model"]

    # log_reg.X_val = X_eval.T
    # log_reg.y_val = y_eval

    # log_reg.train(0, delivery_model["prior"], prior_weighted=True)

    # log_odds, act_dcf, min_dcf = bayes_error_plot(log_reg.llr, y_eval)

    # plot(
    #     {
    #         "actDCF": act_dcf,
    #         "minDCF": min_dcf,
    #     },
    #     log_odds,
    #     file_name="delivery_error_plot",
    #     xlabel="Effective Prior Log Odds",
    #     ylabel="DCF",
    #     marker="",
    # )
