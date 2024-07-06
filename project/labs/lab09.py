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
`numpy.logspace(-5, 0, 11)`. Plot the minDCF and actDCF ( T = 01) as a function of
C (again, use a logarithmic scale for the x-axis). What do you observe? Does the
regularization coefficient significantly affect the results for one or both
metrics (remember that, for SVM, low values of C imply strong regularization,
while large values of C imply weak regularization)? Are the scores well
calibrated for the target application? What can we conclude on linear SVM? How
does it perform compared to other linear models? Repeat the analysis with
centered data. Are the result significantly different?

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

import numpy as np

from project.classifiers.support_vector_machine import SupportVectorMachine
from project.figures.plots import plot
from project.funcs.base import load_data, split_db_2to1
from project.funcs.dcf import dcf
from project.funcs.kernel import poly_kernel, rbf_kernel


def lab09(DATA: str):
    X, y = load_data(DATA)

    (X_train, y_train), (X_val, y_val) = split_db_2to1(X.T, y)

    PRIOR = 0.1

    svm = SupportVectorMachine(X_train, y_train, X_val)

    # Linear SVM with DCF and MinDCF as C varies

    act_dcfs = []
    min_dcfs = []
    range_c = np.logspace(-5, 0, 11)

    for C in range_c:
        scores = svm.train(C, "linear", K=1)

        min_dcfs.append(dcf(scores, y_val, PRIOR, 1.0, 1.0, "min", True))

        act_dcfs.append(dcf(scores, y_val, PRIOR, 1.0, 1.0, "optimal", True))

    plot(
        {
            "minDCF": min_dcfs,
            "actDCF": act_dcfs,
        },
        range_c,
        file_name="svm/linear",
        xscale="log",
        xlabel="C",
        ylabel="DCF",
    )

    # Linear SVM with DCF and MinDCF as C varies with centered data

    act_dcfs = []
    min_dcfs = []
    range_c = np.logspace(-5, 0, 11)

    X_train_centered = X_train - X_train.mean(axis=1, keepdims=True)
    X_val_centered = X_val - X_val.mean(axis=1, keepdims=True)

    svm_centered = SupportVectorMachine(X_train_centered, y_train, X_val_centered)

    for C in range_c:
        scores = svm_centered.train(C, "linear", K=1)

        min_dcfs.append(dcf(scores, y_val, PRIOR, 1.0, 1.0, "min", True))

        act_dcfs.append(dcf(scores, y_val, PRIOR, 1.0, 1.0, "optimal", True))

    plot(
        {
            "minDCF": min_dcfs,
            "actDCF": act_dcfs,
        },
        range_c,
        file_name="svm/linear_centered",
        xscale="log",
        xlabel="C",
        ylabel="DCF",
    )

    # Polynomial Kernel SVM with DCF and minDCF as C varies

    act_dcfs = []
    min_dcfs = []

    for C in range_c:
        scores = svm.train(C, "kernel", K=1, kernel_func=poly_kernel(2, 1))

        min_dcfs.append(dcf(scores, y_val, PRIOR, 1.0, 1.0, "min", True))

        act_dcfs.append(dcf(scores, y_val, PRIOR, 1.0, 1.0, "optimal", True))

    plot(
        {
            "minDCF": min_dcfs,
            "actDCF": act_dcfs,
        },
        range_c,
        file_name="svm/poly_kernel",
        xscale="log",
        xlabel="C",
        ylabel="DCF",
    )

    # RBF Kernel SVM with DCF and minDCF as C varies

    gamma = [
        ("e-4", np.exp(-4)),
        ("e-3", np.exp(-3)),
        ("e-2", np.exp(-2)),
        ("e-1", np.exp(-1)),
    ]
    Cs = np.logspace(-3, 2, 11)

    for l, g in gamma:
        act_dcfs = []
        min_dcfs = []

        for C in Cs:
            scores = svm.train(C, "kernel", K=1, kernel_func=rbf_kernel(g))

            min_dcfs.append(dcf(scores, y_val, PRIOR, 1.0, 1.0, "min", True))

            act_dcfs.append(dcf(scores, y_val, PRIOR, 1.0, 1.0, "optimal", True))

        plot(
            {
                "minDCF": min_dcfs,
                "actDCF": act_dcfs,
            },
            Cs,
            file_name=f"svm/rbf_kernel_{l}",
            xscale="log",
            xlabel="C",
            ylabel="DCF",
        )

    # Optional: Polynomial Kernel SVM with d = 4, c = 1, ξ = 0

    act_dcfs = []
    min_dcfs = []

    for C in range_c:
        scores = svm.train(C, "kernel", K=1, kernel_func=poly_kernel(4, 1))

        min_dcfs.append(dcf(scores, y_val, PRIOR, 1.0, 1.0, "min", True))

        act_dcfs.append(dcf(scores, y_val, PRIOR, 1.0, 1.0, "optimal", True))

    plot(
        {
            "minDCF": min_dcfs,
            "actDCF": act_dcfs,
        },
        range_c,
        file_name="svm/poly_kernel_4",
        xscale="log",
        xlabel="C",
        ylabel="DCF",
    )
