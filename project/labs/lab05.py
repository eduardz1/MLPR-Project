import numpy as np
from rich.console import Console

from project.classifiers.gaussian import Gaussian
from project.figures.plots import heatmap, plot
from project.figures.rich import table
from project.funcs.common import load_data


def lab5(DATA: str):
    console = Console()
    np.set_printoptions(precision=3, suppress=True)

    X, y = load_data(DATA)

    classifier = Gaussian(X, y)

    # Analyze performance of various classifiers with the full dataset

    accuracy, error_rate, llr = classifier.fit(multivariate=True)
    table(
        console,
        "MV Gaussian classifier",
        {
            "Accuracy": [f"{accuracy:.2f}%"],
            "Error rate": [f"{error_rate:.2f}%"],
        },
        llr,
    )

    accuracy, error_rate, _ = classifier.fit(tied=True)
    table(
        console,
        "Tied Covariance Gaussian classifier",
        {
            "Accuracy": [f"{accuracy:.2f}%"],
            "Error rate": [f"{error_rate:.2f}%"],
        },
    )

    accuracy, error_rate, _ = classifier.fit(naive=True)
    table(
        console,
        "Naive Gaussian classifier",
        {
            "Accuracy": [f"{accuracy:.2f}%"],
            "Error rate": [f"{error_rate:.2f}%"],
        },
    )

    # Display the covariance matrix of each class

    covariances = classifier.covariances
    table(
        console,
        "Covariance matrices",
        {
            "Fake": [f"{covariances[0]}"],
            "Genuine": [f"{covariances[1]}"],
        },
    )

    # Display the correlation matrices of each class

    correlation_matrices = classifier.corrcoef
    table(
        console,
        "Correlation matrices",
        {
            "Fake": [f"{correlation_matrices[0]}"],
            "Genuine": [f"{correlation_matrices[1]}"],
        },
    )

    # Plot covariances and correlation matrices as heatmaps

    heatmap(covariances[0], "Reds", "covariance_fake")
    heatmap(covariances[1], "Blues", "covariance_genuine")
    heatmap(correlation_matrices[0], "Reds", "correlation_fake")
    heatmap(correlation_matrices[1], "Blues", "correlation_genuine")

    # Try again repeating the classification without the last two features,
    # which do not fit well with the Gaussian assumption

    accuracy, error_rate, llr = classifier.fit(multivariate=True, slice=slice(-2))
    table(
        console,
        "MV Gaussian classifier (without last two features)",
        {
            "Accuracy": [f"{accuracy:.2f}%"],
            "Error rate": [f"{error_rate:.2f}%"],
        },
        llr,
    )

    accuracy, error_rate, _ = classifier.fit(tied=True, slice=slice(-2))
    table(
        console,
        "Tied Covariance Gaussian classifier (without last two features)",
        {
            "Accuracy": [f"{accuracy:.2f}%"],
            "Error rate": [f"{error_rate:.2f}%"],
        },
    )

    accuracy, error_rate, _ = classifier.fit(naive=True, slice=slice(-2))
    table(
        console,
        "Naive Gaussian classifier (without last two features)",
        {
            "Accuracy": [f"{accuracy:.2f}%"],
            "Error rate": [f"{error_rate:.2f}%"],
        },
    )

    # Benchmark MVG and Tied Covariance Gaussian classifiers for first two features

    accuracy, error_rate, llr = classifier.fit(multivariate=True, slice=slice(2))
    table(
        console,
        "MV Gaussian classifier (first two features)",
        {
            "Accuracy": [f"{accuracy:.2f}%"],
            "Error rate": [f"{error_rate:.2f}%"],
        },
        llr,
    )

    accuracy, error_rate, _ = classifier.fit(tied=True, slice=slice(2))
    table(
        console,
        "Tied Covariance Gaussian classifier (first two features)",
        {
            "Accuracy": [f"{accuracy:.2f}%"],
            "Error rate": [f"{error_rate:.2f}%"],
        },
    )

    # Benchmark MVG and Tied Covariance Gaussian classifiers for third and fourth features

    accuracy, error_rate, llr = classifier.fit(multivariate=True, slice=slice(2, 4))
    table(
        console,
        "MV Gaussian classifier (third and fourth features)",
        {
            "Accuracy": [f"{accuracy:.2f}%"],
            "Error rate": [f"{error_rate:.2f}%"],
        },
        llr,
    )

    accuracy, error_rate, _ = classifier.fit(tied=True, slice=slice(2, 4))
    table(
        console,
        "Tied Covariance Gaussian classifier (third and fourth features)",
        {
            "Accuracy": [f"{accuracy:.2f}%"],
            "Error rate": [f"{error_rate:.2f}%"],
        },
    )

    # Try to reduce the dimensionality with PCA

    accuracies_mvg = []
    accuracies_tied = []
    accuracies_naive = []

    for i in range(1, 7):
        accuracy, _, _ = classifier.fit(multivariate=True, pca=True, pca_dimensions=i)
        accuracies_mvg.append(accuracy)

        accuracy, _, _ = classifier.fit(tied=True, pca=True, pca_dimensions=i)
        accuracies_tied.append(accuracy)

        accuracy, _, _ = classifier.fit(naive=True, pca=True, pca_dimensions=i)
        accuracies_naive.append(accuracy)

    plot(
        {
            "Multivariate": accuracies_mvg,
            "Tied Covariance": accuracies_tied,
            "Naive Bayes": accuracies_naive,
        },
        range(1, 7),
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
