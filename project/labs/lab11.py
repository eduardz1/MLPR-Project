"""
# Calibration and Evaluation

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
import math
import os
from functools import partial
from typing import Literal

import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from project.classifiers.gaussian_mixture_model import GaussianMixtureModel
from project.classifiers.logistic_regression import LogisticRegression
from project.classifiers.support_vector_machine import SupportVectorMachine
from project.figures.plots import plot
from project.figures.rich import table
from project.funcs.base import load_data, quadratic_feature_expansion, split_db_2to1
from project.funcs.dcf import bayes_error, dcf
from project.funcs.kfolds import kfolds

TARGET_PRIOR = 0.1

NUM_POINTS_BAYES_ERROR_PLOT = 100

K_FOLDS = 5


def find_best_prior_and_plot_bayes_error(
    scores,
    priors: npt.NDArray,
    y_val: npt.NDArray,
    model: Literal["log_reg", "svm", "gmm", "fusion"] = "fusion",
    fusion: bool = False,
):
    if not fusion:
        min_dcf_target = dcf(scores[0], y_val, TARGET_PRIOR, "min").item()
        log_odds, act_dcf_applications, min_dcf_applications = bayes_error(
            scores[0], y_val, num_points=NUM_POINTS_BAYES_ERROR_PLOT
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
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Fusing models..." if fusion else f"Calibrating {model}...",
            total=len(priors) * K_FOLDS,
        )

        for pi in priors:
            calibrated_scores, calibrated_labels = kfolds(
                scores,
                y_val,
                partial(progress.update, task_id=task, advance=1),
                pi,
                K_FOLDS,
            )

            act_dcf = dcf(
                calibrated_scores, calibrated_labels, TARGET_PRIOR, "optimal"
            ).item()

            if act_dcf < best_act_dcf_target:
                best_act_dcf_target = act_dcf
                best_prior = pi
                best_calibrated_scores = calibrated_scores
                best_calibrated_labels = calibrated_labels

        log_odds, act_dcf_applications, min_dcf_applications = bayes_error(
            best_calibrated_scores,
            best_calibrated_labels,
            num_points=NUM_POINTS_BAYES_ERROR_PLOT,
        )

        if fusion:
            applications["actDCF (fused scores)"] = act_dcf_applications.tolist()
            applications["minDCF (fused scores)"] = min_dcf_applications.tolist()
        else:
            applications["actDCF (post calibration)"] = act_dcf_applications.tolist()

        progress.console.print(f"[cyan]Best prior for {model}: {best_prior:.1f}")

        plot(
            applications,
            log_odds,
            colors=["purple", "purple"] if fusion else ["purple", "purple", "purple"],
            linestyles=["dashed", "solid"] if fusion else ["dotted", "dashed", "solid"],
            file_name=f"calibration/{model}",
            xlabel="Effective Prior Log Odds",
            ylabel="DCF",
            marker="",
        )

        return (
            best_prior,
            best_act_dcf_target,
            None if fusion else min_dcf_target,
            applications,
            log_odds,
        )


def lab11(DATA: str):
    console = Console()

    np.set_printoptions(precision=3, suppress=True)

    X, y = load_data(DATA)
    (X_eval, y_eval) = load_data("data/evalData.txt")

    (X_train, y_train), (_, y_val) = split_db_2to1(X.T, y)

    priors = np.linspace(0.1, 0.9, 9)

    # region Load models and scores

    try:
        f = open("scores/log_reg.npy", "rb")
        g = open("models/log_reg.json", "r")
    except FileNotFoundError:
        console.print("[red]Please run lab8 first.")
        return
    else:
        with f:
            BEST_LOG_REG_SCORES = np.load(f, allow_pickle=True)
        with g:
            LOG_REG = LogisticRegression.from_json(json.load(g))
    try:
        f = open("scores/svm.npy", "rb")
        g = open("models/svm.json", "r")
    except FileNotFoundError:
        console.print("[red]Please run lab9 first.")
        return
    else:
        with f:
            BEST_SVM_SCORES = np.load(f, allow_pickle=True)
        with g:
            SVM = SupportVectorMachine.from_json(json.load(g))
    try:
        f = open("scores/gmm.npy", "rb")
        g = open("models/gmm.json", "r")

        files = [file for file in os.listdir("models") if file.startswith("_")]

        ALL_GMM = {}
        for file in files:
            with open(os.path.join("models", file), "r") as h:
                ALL_GMM[file] = h.read()

        ALL_GMM = {
            k: GaussianMixtureModel.from_json(json.loads(v)) for k, v in ALL_GMM.items()
        }
    except FileNotFoundError:
        console.print("[red]Please run lab10 first.")
        return
    else:
        with f:
            BEST_GMM_SCORES = np.load(f, allow_pickle=True)
        with g:
            GMM = GaussianMixtureModel.from_json(json.load(g))

    LOG_REG.X_train = (
        quadratic_feature_expansion(X_train) if LOG_REG._quadratic else X_train
    )
    LOG_REG.y_train = y_train
    LOG_REG.X_val = (
        quadratic_feature_expansion(X_eval.T) if LOG_REG._quadratic else X_eval.T
    )
    LOG_REG.y_val = y_eval

    SVM.X_train = X_train
    SVM.y_train = y_train
    SVM.X_val = X_eval.T
    SVM.y_val = y_eval

    GMM.X_train = X_train
    GMM.y_train = y_train
    GMM.X_val = X_eval.T
    GMM.y_val = y_eval

    models = {
        "log_reg": ([LOG_REG], [BEST_LOG_REG_SCORES]),
        "svm": ([SVM], [BEST_SVM_SCORES]),
        "gmm": ([GMM], [BEST_GMM_SCORES]),
        "fusion": (
            [LOG_REG, SVM, GMM],
            [BEST_LOG_REG_SCORES, BEST_SVM_SCORES, BEST_GMM_SCORES],
        ),
    }

    # region Calibration

    stats = {
        "log_reg": {
            "prior": None,
            "actDCF": np.inf,
            "minDCF": np.inf,
            "applications": [],
        },
        "svm": {
            "prior": None,
            "actDCF": np.inf,
            "minDCF": np.inf,
            "applications": [],
        },
        "gmm": {
            "prior": None,
            "actDCF": np.inf,
            "minDCF": np.inf,
            "applications": [],
        },
        "fusion": {
            "prior": None,
            "actDCF": np.inf,
            "applications": [],
        },
    }

    for k, (_, scores) in models.items():
        pi, act_dcf, min_dcf, apps, log_odds = find_best_prior_and_plot_bayes_error(
            scores, priors, y_val, k, k == "fusion"  # type: ignore
        )
        stats[k]["actDCF"] = act_dcf
        if k != "fusion":
            stats[k]["minDCF"] = min_dcf
        stats[k]["prior"] = pi
        stats[k]["applications"] = apps

    # # table(console, "Best Model after Calibration", delivery_model)

    delivery_model = min(stats.items(), key=lambda item: item[1]["actDCF"])

    table(
        console,
        "Best Model after Calibration",
        {
            "Model": delivery_model[0],
            "actDCF": delivery_model[1]["actDCF"],
            "minDCF": delivery_model[1]["minDCF"],
            "Prior": delivery_model[1]["prior"],
        },
    )

    # Plot the combination of Bayes Error Plots
    plot(
        {
            "LogReg - actDCF (post calibration)": stats["log_reg"]["applications"][
                "actDCF (post calibration)"
            ],
            "LogReg - actDCF (pre calibration)": stats["log_reg"]["applications"][
                "actDCF (pre calibration)"
            ],
            "LogReg - minDCF": stats["log_reg"]["applications"]["minDCF"],
            "SVM - actDCF (post calibration)": stats["svm"]["applications"][
                "actDCF (post calibration)"
            ],
            "SVM - actDCF (pre calibration)": stats["svm"]["applications"][
                "actDCF (pre calibration)"
            ],
            "SVM - minDCF": stats["svm"]["applications"]["minDCF"],
            "GMM - actDCF (post calibration)": stats["gmm"]["applications"][
                "actDCF (post calibration)"
            ],
            "GMM - actDCF (pre calibration)": stats["gmm"]["applications"][
                "actDCF (pre calibration)"
            ],
            "GMM - minDCF": stats["gmm"]["applications"]["minDCF"],
            "Fusion - actDCF": stats["fusion"]["applications"]["actDCF (fused scores)"],
            "Fusion - minDCF": stats["fusion"]["applications"]["minDCF (fused scores)"],
        },
        log_odds,
        colors=[
            "orange",
            "orange",
            "orange",
            "purple",
            "purple",
            "purple",
            "green",
            "green",
            "green",
            "pink",
            "pink",
        ],
        linestyles=["-", ":", "--", "-", ":", "--", "-", ":", "--", "-", "--"],
        file_name="calibration/complete",
        xlabel="Effective prior log odds",
        ylabel="DCF",
        marker="",
        figsize=(10, 7.5),
    )

    # region Evaluation

    for k, (model, scores) in models.items():
        SCAL = np.vstack(scores)
        SVAL = np.vstack([m.llr for m in model])

        cl = LogisticRegression(SCAL, y_val, SVAL, y_eval)

        cl.train(0, stats[k]["prior"])

        cal_scores = np.hstack([cl.llr])
        cal_labels = np.hstack([y_eval])

        log_odds, act_dcf, min_dcf = bayes_error(cal_scores, cal_labels)

        stats[k]["actDCF"] = dcf(cal_scores, cal_labels, TARGET_PRIOR, "optimal").item()
        stats[k]["minDCF"] = dcf(cal_scores, cal_labels, TARGET_PRIOR, "min").item()
        stats[k]["applications"] = {
            "actDCF": act_dcf.tolist(),
            "minDCF": min_dcf.tolist(),
        }

    # 1 - Plot Bayes Error Plot for our delivery model

    plot(
        stats[delivery_model[0]]["applications"],
        log_odds,
        colors=["purple", "purple"],
        linestyles=["-", "--"],
        file_name="evaluation/delivery",
        xlabel="Effective Prior Log Odds",
        ylabel="DCF",
        marker="",
    )

    # 2 - Evaluate the four models and plot the Bayes Error Plots for actual DCF

    plot(
        {
            "LogReg - actDCF": stats["log_reg"]["applications"]["actDCF"],
            "SVM - actDCF": stats["svm"]["applications"]["actDCF"],
            "GMM - actDCF": stats["gmm"]["applications"]["actDCF"],
            "Fusion - actDCF": stats["fusion"]["applications"]["actDCF"],
        },
        log_odds,
        colors=["orange", "purple", "green", "pink"],
        file_name="evaluation/actDCF",
        xlabel="Effective Prior Log Odds",
        ylabel="DCF",
        marker="",
    )

    # 3 - Plot Bayes Error plot for the three systems for actual and min DCF

    plot(
        {
            "LogReg - actDCF": stats["log_reg"]["applications"]["actDCF"],
            "LogReg - minDCF": stats["log_reg"]["applications"]["minDCF"],
            "SVM - actDCF": stats["svm"]["applications"]["actDCF"],
            "SVM - minDCF": stats["svm"]["applications"]["minDCF"],
            "GMM - actDCF": stats["gmm"]["applications"]["actDCF"],
            "GMM - minDCF": stats["gmm"]["applications"]["minDCF"],
        },
        log_odds,
        colors=["orange", "orange", "purple", "purple", "green", "green"],
        linestyles=["-", "--", "-", "--", "-", "--"],
        file_name="evaluation/act_min_DCF",
        xlabel="Effective Prior Log Odds",
        ylabel="DCF",
        marker="",
    )

    table(
        console,
        "minDCF and actDCF for the three models",
        {
            "DCF": ["minimum", "actual"],
            "LogReg": [stats["log_reg"]["minDCF"], stats["log_reg"]["actDCF"]],
            "SVM": [stats["svm"]["minDCF"], stats["svm"]["actDCF"]],
            "GMM": [stats["gmm"]["minDCF"], stats["gmm"]["actDCF"]],
        },
    )

    # 4 - Evaluate all combinations of GMM again

    full = np.empty((6, 6))
    diag = np.empty((6, 6))

    for k, gmm in ALL_GMM.items():
        gmm.X_val = X_eval.T

        llr = gmm.llr

        min_dcf = dcf(llr, y_eval, TARGET_PRIOR, "min").item()

        true = int(math.log2(int(k[7:9])))
        false = int(math.log2(int(k[11:13])))

        if k[1] == "d":
            diag[true][false] = min_dcf
        else:
            full[true][false] = min_dcf

    table(
        console,
        "Comparison with other GMMs (true x false)",
        {"Full": full, "Diagonal": diag},
    )
