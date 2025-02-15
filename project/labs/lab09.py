"""
# Support Vector Machines classification

NOTE: Training SVM models may require some time. To speed-up the process, we
suggest that you first setup the experiments for this project using only a
fraction of the model training data. Once the code is ready, you can then re-run
all the experiments with the full dataset.

Apply the SVM to the project data. Start with the linear model (to avoid
excessive training time we consider only the models trained with K = 1.0). Train
the model with different values of C. As for logistic regression, you should
employ a logarithmic scale for the values of C. Reasonable values are given by
`numpy.logspace(-5, 0, 11)`. Plot the minDCF and actDCF (pi_T = 0.1) as a
function of C (again, use a logarithmic scale for the x-axis). What do you
observe? Does the regularization coefficient significantly affect the results
for one or both metrics (remember that, for SVM, low values of C imply strong
regularization, while large values of C imply weak regularization)? Are the
scores well calibrated for the target application? What can we conclude on
linear SVM? How does it perform compared to other linear models? Repeat the
analysis with centered data. Are the result significantly different?

We now consider the polynomial kernel. For simplicity, we consider only the
kernel with d = 2, c = 1 (but better results may be possible with different
configurations), and we set ξ = 0, since the kernel already implicitly accounts
for the bias term (due to c = 1). We also consider only the original,
non-centered features (again, different pre-processing strategies may lead to
better results). Train the model with different values of C, and compare the
results in terms of minDCF and actDCF. What do you observe with quadratic
models? In light of the characteristics of the dataset and of the classifier,
are the results consistent with previous models (logistic regression and MVG
models) in terms of minDCF? What about actDCF?

For RBF kernel we need to optimize both γ and C (since the RBF kernel does not
implicitly account for the bias term we set ξ = 1). We adopt a grid search
approach, i.e., we consider different values of γ and different values of C, and
try all possible combinations. For we suggest you analyze values γ ∈ [e^(-4),
e^(-3), e^(-2), e^(-1)], while for C, to avoid excessive time but obtain a good
coverage of possible good values we suggest log-spaced values
`numpy.logspace(-3, 2, 11)` (of course, you are free to experiment with other
values if you so wish). Train all models obtained by combining the values of γ
and of C. Plot minDCF and actDCF as a function of C, with a different line for
each value of γ (i.e., four lines for minDCF and four lines for actDCF). Analyze
the results. Are there values of γ and C that provide better results? Are the
scores well calibrated? How the result compare to previous models? Are there
characteristics of the dataset that can be better captured by RBF kernels?

## Optional

Consider again the polynomial kernel, but with d = 4, c = 1, ξ = 0. Train the
model with different values of C (use again `numpy.logspace(-5, 0, 11)`, and
compare the results in terms of minDCF and actDCF. What do you observe with
quadratic models? Can you explain the better results in terms of the
characteristics of the dataset? To answer, consider only the last two features
of each sample (look at the scatter plots) yi = xi[4:6]. Consider how these
features would be transformed by a simple degree 2 kernel that maps each sample
to yi -> zi = yi[0:1]yi[1:2] (suggestion: draw few samples on paper, one per
cluster). Then consider which kind of separation surfaces would be required to
separate the zi 1-D feature vectors — remember that in 1-D linear rules
correspond to a sample being on either side of a threshold, quadratic rules
correspond to inequalities that involve the sample being inside a (possibly
empty) interval, and so on.
"""

import json

import numpy as np
from rich.console import Console
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn

from project.classifiers.support_vector_machine import SupportVectorMachine
from project.figures.plots import plot
from project.figures.rich import table
from project.funcs.base import load_data, split_db_2to1
from project.funcs.dcf import dcf


def lab09(DATA: str):
    console = Console()

    X, y = load_data(DATA)

    (X_train, y_train), (X_val, y_val) = split_db_2to1(X.T, y)

    PRIOR = 0.1

    linear_svm = SupportVectorMachine("linear")
    poly_svm = SupportVectorMachine("poly_kernel")
    rbf_svm = SupportVectorMachine("rbf_kernel")

    range_c = np.logspace(-5, 0, 11)

    best_svm_config = {
        "svm_type": "",
        "min_dcf": np.inf,
        "act_dcf": np.inf,
        "C": 0,
        "kernel_func": None,
        "centered": False,
        "scores": None,
        "model": None,
    }

    # Linear SVM with DCF and MinDCF as C varies

    act_dcfs = []
    min_dcfs = []

    with Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Linear SVMs", total=len(range_c))

        for C in range_c:
            scores = linear_svm.fit(X_train, y_train, K=1, C=C).scores(X_val)
            min_dcfs.append(dcf(scores, y_val, PRIOR, "min").item())
            act_dcfs.append(dcf(scores, y_val, PRIOR, "optimal").item())

            if min_dcfs[-1] < best_svm_config["min_dcf"]:
                best_svm_config.update(
                    {
                        "svm_type": "linear",
                        "min_dcf": min_dcfs[-1],
                        "act_dcf": act_dcfs[-1],
                        "C": C,
                        "kernel_func": None,
                        "centered": False,
                        "scores": scores,
                        "model": linear_svm.to_json(),
                    }
                )

            progress.update(task, advance=1)

    min_dcfs_non_centered = min_dcfs.copy()
    act_dcfs_non_centered = act_dcfs.copy()

    # Linear SVM with DCF and MinDCF as C varies with centered data

    act_dcfs = []
    min_dcfs = []

    X_train_centered = X_train - X_train.mean(axis=1, keepdims=True)
    X_val_centered = X_val - X_val.mean(axis=1, keepdims=True)

    with Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Linear SVMs with centered data",
            total=len(range_c),
        )

        for C in range_c:
            scores = linear_svm.fit(X_train_centered, y_train, K=1, C=C).scores(
                X_val_centered
            )
            min_dcfs.append(dcf(scores, y_val, PRIOR, "min").item())
            act_dcfs.append(dcf(scores, y_val, PRIOR, "optimal").item())

            if min_dcfs[-1] < best_svm_config["min_dcf"]:
                best_svm_config.update(
                    {
                        "svm_type": "linear",
                        "min_dcf": min_dcfs[-1],
                        "act_dcf": act_dcfs[-1],
                        "C": C,
                        "kernel_func": None,
                        "centered": True,
                        "scores": scores,
                        "model": linear_svm.to_json(),
                    }
                )

            progress.update(task, advance=1)

    plot(
        {
            "minDCF (centered)": min_dcfs,
            "minDCF": min_dcfs_non_centered,
            "actDCF (centered)": act_dcfs,
            "actDCF": act_dcfs_non_centered,
        },
        range_c,
        colors=["yellowgreen", "green", "violet", "purple"],
        linestyles=["dotted", "dashed", "-.", "solid"],
        file_name="svm/linear",
        xscale="log",
        xlabel="C",
        ylabel="DCF",
    )

    # Polynomial Kernel SVM with DCF and minDCF as C varies

    act_dcfs = []
    min_dcfs = []

    with Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Polynomial Kernel SVMs of degree 2",
            total=len(range_c),
        )

        for C in range_c:
            scores = poly_svm.fit(
                X_train, y_train, K=1, C=C, degree=2, c=1, xi=0
            ).scores(X_val)
            min_dcfs.append(dcf(scores, y_val, PRIOR, "min").item())
            act_dcfs.append(dcf(scores, y_val, PRIOR, "optimal").item())

            if min_dcfs[-1] < best_svm_config["min_dcf"]:
                best_svm_config.update(
                    {
                        "min_dcf": min_dcfs[-1],
                        "act_dcf": act_dcfs[-1],
                        "C": C,
                        "svm_type": "kernel",
                        "kernel_func": "poly_kernel(2, 1)",
                        "centered": False,
                        "scores": scores,
                        "model": poly_svm.to_json(),
                    }
                )

            progress.update(task, advance=1)

    min_dcfs_poly_2 = min_dcfs.copy()
    act_dcfs_poly_2 = act_dcfs.copy()

    # RBF Kernel SVM with DCF and minDCF as C varies

    gamma = [
        ("$\\gamma = e^{-4}$", np.exp(-4)),
        ("$\\gamma = e^{-3}$", np.exp(-3)),
        ("$\\gamma = e^{-2}$", np.exp(-2)),
        ("$\\gamma = e^{-1}$", np.exp(-1)),
    ]
    Cs = np.logspace(-3, 2, 11)

    with Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "RBF Kernel SVMs",
            total=len(Cs) * len(gamma),
        )

        act_dcfs_gamma = []
        min_dcfs_gamma = []

        for _, g in gamma:

            act_dcfs = []
            min_dcfs = []

            for C in Cs:
                scores = rbf_svm.fit(X_train, y_train, C=C, K=1, gamma=g).scores(X_val)
                min_dcfs.append(dcf(scores, y_val, PRIOR, "min").item())
                act_dcfs.append(dcf(scores, y_val, PRIOR, "optimal").item())

                if min_dcfs[-1] < best_svm_config["min_dcf"]:
                    best_svm_config.update(
                        {
                            "min_dcf": min_dcfs[-1],
                            "act_dcf": act_dcfs[-1],
                            "C": C,
                            "svm_type": "kernel",
                            "kernel_func": f"rbf_kernel({g})",
                            "centered": False,
                            "scores": scores,
                            "model": rbf_svm.to_json(),
                        }
                    )

                progress.update(task, advance=1)

            act_dcfs_gamma.append(act_dcfs)
            min_dcfs_gamma.append(min_dcfs)

        plot(
            {
                **{
                    f"minDCF {l}": min_dcf
                    for (l, _), min_dcf in zip(gamma, min_dcfs_gamma)  # noqa: E741
                },
                **{
                    f"actDCF {l}": act_dcf
                    for (l, _), act_dcf in zip(gamma, act_dcfs_gamma)  # noqa: E741
                },
            },
            Cs,
            colors=[f"C{i % len(gamma)}" for i in range(len(gamma) * 2)],
            linestyles=[
                "dashed" if i < len(gamma) else "solid" for i in range(len(gamma) * 2)
            ],
            file_name="svm/rbf_kernel",
            xscale="log",
            xlabel="C",
            ylabel="DCF",
        )

    # Optional: Polynomial Kernel SVM with d = 4, c = 1, ξ = 0

    act_dcfs = []
    min_dcfs = []

    with Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Polynomial Kernel SVMs of degree 4",
            total=len(range_c),
        )

        for C in range_c:
            scores = poly_svm.fit(
                X_train, y_train, K=1, C=C, degree=4, c=1, xi=0
            ).scores(X_val)
            min_dcfs.append(dcf(scores, y_val, PRIOR, "min").item())
            act_dcfs.append(dcf(scores, y_val, PRIOR, "optimal").item())

            if min_dcfs[-1] < best_svm_config["min_dcf"]:
                best_svm_config.update(
                    {
                        "min_dcf": min_dcfs[-1],
                        "act_dcf": act_dcfs[-1],
                        "C": C,
                        "svm_type": "kernel",
                        "kernel_func": "poly_kernel(4, 1)",
                        "centered": False,
                        "scores": scores,
                        "model": poly_svm.to_json(),
                    }
                )

            progress.update(task, advance=1)

    plot(
        {
            "minDCF $d = 4$": min_dcfs,
            "minDCF $d = 2$": min_dcfs_poly_2,
            "actDCF $d = 4$": act_dcfs,
            "actDCF $d = 2$": act_dcfs_poly_2,
        },
        range_c,
        colors=["yellowgreen", "green", "violet", "purple"],
        linestyles=["dotted", "dashed", "-.", "solid"],
        file_name="svm/poly_kernel",
        xscale="log",
        xlabel="C",
        ylabel="DCF",
    )

    with open("models/scores/svm.npy", "wb") as f:
        np.save(f, best_svm_config["scores"])

    with open("models/svm.json", "w") as f:
        json.dump(best_svm_config["model"], f, indent=4)

    best_svm_config.pop("scores")
    best_svm_config.pop("model")
    table(console, "Best SVM Configuration", best_svm_config)
