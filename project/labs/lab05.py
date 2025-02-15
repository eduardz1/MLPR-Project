"""
# Generative models for classification

Apply the MVG model to the project data. Split the dataset in model training and
validation subsets (important: use the same splits for all models, including
those presented in other laboratories), train the model parameters on the model
training portion of the dataset and compute LLRs

`s(xt) = llr(xt) = fX|C (xt|1) / fX|C (xt|0)`

(i.e., with class True, label 1 on top of the ratio) for the validation subset.
Obtain predictions from LLRs assuming uniform class priors
P (C = 1) = P (C = 0) = 1/2. Compute the corresponding error rate (suggestion:
in the next laboratories we will modify the way we compute predictions from
LLRs, we therefore recommend that you keep separated the functions that compute
LLRs, those that compute predictions from LLRs and those that compute error rate
from predictions).

Apply now the tied Gaussian model, and compare the results with MVG and LDA.
Which model seems to perform better?

Finally, test the Naive Bayes Gaussian model. How does it compare with the
previous two?

Let’s now analyze the results in light of the characteristics of the features
that we observed in previous laboratories. Start by printing the covariance
matrix of each class (you can extract this from the MVG model parameters). The
covariance matrices contain, on the diagonal, the variances for the different
features, whereas the elements outside of the diagonal are the feature
co-variances. For each class, compare the covariance of different feature pairs
with the respective variances. What do you observe? Are co-variance values large
or small compared to variances? To better visualize the strength of co-variances
with respect to variances we can compute, for a pair of features i, j, the
Pearson correlation coefficient, defined as

`Corr(i, j) = Cov(i, j) / (√V ar(i) √V ar(j))`

or, directly matrix form,

`Corr = C / ( vcol(C.diagonal()**0.5) * vrow(C.diagonal()**0.5) )`

where C is a covariance matrix. The correlation matrix has diagonal elements
equal to 1, whereas out-of-diagonal elements correspond to the correlation
coefficients for all feature pairs, with −1 ≤ Corr(i, j) ≤ 1. When
Corr(i, j) = 0 the features i, j are uncorrelated, whereas values close to ±1
denote strong correlation.

Compute the correlation matrices for the two classes. What can you conclude on
the features? Are the features strongly or weakly correlated? How is this
related to the Naive Bayes results?

The Gaussian model assumes that features can be jointly modeled by Gaussian
distributions. The goodness of the model is therefore strongly affected by the
accuracy of this assumption. Although visualizing 6-dimensional distributions is
unfeasible, we can analyze how well the assumption holds for single (or pairs)
of features. In Laboratory 4 we separately fitted a Gaussian density over each
feature for each class. This corresponds to the Naive Bayes model. What can you
conclude on the goodness of the Gaussian assumption? Is it accurate for all the
6 features? Are there features for which the assumptions do not look good?

To analyze if indeed the last set of features negatively affects our classifier
because of poor modeling assumptions, we can try repeating the classification
using only feature 1 to 4 (i.e., discarding the last 2 features). Repeat the
analysis for the three models. What do you obtain? What can we conclude on
discarding the last two features? Despite the inaccuracy of the assumption for
these two features, are the Gaussian models still able to extract some useful
information to improve classification accuracy?

In Laboratory 2 and 4 we analyzed the distribution of features 1-2 and of
features 3-4, finding that for features 1 and 2 means are similar but variances
are not, whereas for features 3 and 4 the two classes mainly differ for the
feature mean, but show similar variance. Furthermore, the features also show
limited correlation for both classes. We can analyze how these characteristics
of the features distribution affect the performance of the different approaches.
Repeat the classification using only features 1-2 (jointly), and then do the
same using only features 3-4 (jointly), and compare the results of the MVG and
tied MVG models. In the first case, which model is better? And in the second
case? How is this related to the characteristics of the two classifiers? Is the
tied model effective at all for the first two features? Why? And the MVG? And
for the second pair of features?

Finally, we can analyze the effects of PCA as pre-processing. Use PCA to reduce
the dimensionality of the feature space, and apply the three classification
approaches. What do you observe? Is PCA effective for this dataset with the
Gaussian models? Overall, what is the model that provided the best accuracy
on the validation set?
"""

import numpy as np
from rich.console import Console

from project.classifiers.binary_gaussian import BinaryGaussian
from project.figures.plots import heatmap, plot
from project.figures.rich import table
from project.funcs.base import load_data, split_db_2to1
from project.funcs.dcf import dcf


def lab05(DATA: str):
    console = Console()

    np.set_printoptions(precision=3, suppress=True)

    X, y = load_data(DATA)

    (X_train, y_train), (X_val, y_val) = split_db_2to1(X.T, y)

    X_train = X_train.T
    X_val = X_val.T

    mvg = BinaryGaussian("multivariate")
    tied = BinaryGaussian("tied")
    naive = BinaryGaussian("naive")

    stats = {
        "Accuracy": "",
        "Error rate": "",
    }

    PRIOR = 0.1

    best_model_min_dcf = {"min_dcf": np.inf}

    def save_stats(cl, best_model_min_dcf):
        stats["Accuracy"] = f"{cl.accuracy:.2f}%"
        stats["Error rate"] = f"{cl.error_rate:.2f}%"

        min_dcf = dcf(cl.llr, y_val, PRIOR, "min").item()
        if min_dcf < best_model_min_dcf["min_dcf"]:
            best_model_min_dcf["min_dcf"] = min_dcf

            with open("models/scores/bin_gau.npy", "wb") as f:
                np.save(f, cl.llr)

            with open("models/bin_gau.json", "w") as f:
                cl.to_json(f)

    # Analyze performance of various classifiers with the full dataset

    mvg.fit(X_train, y_train).predict(X_val, y_val)
    save_stats(mvg, best_model_min_dcf)
    table(console, "MV Gaussian classifier", stats, mvg.llr)

    tied.fit(X_train, y_train).predict(X_val, y_val)
    save_stats(tied, best_model_min_dcf)
    table(console, "Tied Covariance Gaussian classifier", stats, tied.llr)

    naive.fit(X_train, y_train).predict(X_val, y_val)
    save_stats(naive, best_model_min_dcf)
    table(console, "Naive Gaussian classifier", stats, naive.llr)

    # Display the covariance matrix of each class

    table(
        console,
        "Covariance matrices",
        {
            "Fake": f"{mvg.C[0]}",
            "Genuine": f"{mvg.C[1]}",
        },
    )

    # Display the correlation matrices of each class

    table(
        console,
        "Correlation matrices",
        {
            "Fake": f"{mvg.corr[0]}",
            "Genuine": f"{mvg.corr[1]}",
        },
    )

    # Plot covariances and correlation matrices as heatmaps

    heatmap(mvg.C[0], "Reds", "covariance_fake")
    heatmap(mvg.C[1], "Blues", "covariance_genuine")
    heatmap(mvg.corr[0], "Reds", "correlation_fake")
    heatmap(mvg.corr[1], "Blues", "correlation_genuine")

    # Try again repeating the classification without the last two features,
    # which do not fit well with the Gaussian assumption

    mvg.fit(X_train, y_train, slicer=slice(-2)).predict(X_val, y_val)
    save_stats(mvg, best_model_min_dcf)
    table(
        console,
        "MV Gaussian classifier (without last two features)",
        stats,
        mvg.llr,
    )

    tied.fit(X_train, y_train, slicer=slice(-2)).predict(X_val, y_val)
    save_stats(tied, best_model_min_dcf)
    table(
        console,
        "Tied Covariance Gaussian classifier (without last two features)",
        stats,
        tied.llr,
    )

    naive.fit(X_train, y_train, slicer=slice(-2)).predict(X_val, y_val)
    save_stats(naive, best_model_min_dcf)
    table(
        console,
        "Naive Gaussian classifier (without last two features)",
        stats,
        naive.llr,
    )

    # Benchmark MVG and Tied Covariance Gaussian classifiers for first two features

    mvg.fit(X_train, y_train, slicer=slice(2)).predict(X_val, y_val)
    save_stats(mvg, best_model_min_dcf)
    table(
        console,
        "MV Gaussian classifier (first two features)",
        stats,
        mvg.llr,
    )

    tied.fit(X_train, y_train, slicer=slice(2)).predict(X_val, y_val)
    save_stats(tied, best_model_min_dcf)
    table(
        console,
        "Tied Covariance Gaussian classifier (first two features)",
        stats,
        tied.llr,
    )

    # Benchmark MVG and Tied Covariance Gaussian classifiers for third and fourth features

    mvg.fit(X_train, y_train, slicer=slice(2, 4)).predict(X_val, y_val)
    save_stats(mvg, best_model_min_dcf)
    table(
        console,
        "MV Gaussian classifier (third and fourth features)",
        stats,
        mvg.llr,
    )

    tied.fit(X_train, y_train, slicer=slice(2, 4)).predict(X_val, y_val)
    save_stats(tied, best_model_min_dcf)
    table(
        console,
        "Tied Covariance Gaussian classifier (third and fourth features)",
        stats,
        tied.llr,
    )

    # Try to reduce the dimensionality with PCA

    accuracies_mvg = []
    accuracies_tied = []
    accuracies_naive = []

    for i in range(1, 7):
        mvg.fit(X_train, y_train, pca_dims=i).predict(X_val, y_val)
        accuracies_mvg.append(mvg.accuracy)

        tied.fit(X_train, y_train, pca_dims=i).predict(X_val, y_val)
        accuracies_tied.append(tied.accuracy)

        naive.fit(X_train, y_train, pca_dims=i).predict(X_val, y_val)
        accuracies_naive.append(naive.accuracy)

    plot(
        {
            "Multivariate": accuracies_mvg,
            "Tied Covariance": accuracies_tied,
            "Naive Bayes": accuracies_naive,
        },
        range(1, 7),
        colors=["purple", "green", "orange"],
        xlabel="Number of features",
        ylabel="Accuracy (%)",
        file_name="pca_to_gaussians",
        figsize=(8, 4),
    )

    table(
        console,
        "PCA to Gaussian classifiers",
        {
            "Number of features": [*range(1, 7)],
            "Multivariate": [f"{a:.2f}%" for a in accuracies_mvg],
            "Tied Covariance": [f"{a:.2f}%" for a in accuracies_tied],
            "Naive Bayes": [f"{a:.2f}%" for a in accuracies_naive],
        },
    )
