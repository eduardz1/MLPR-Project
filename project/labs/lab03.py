import numpy as np
from rich.console import Console
from sklearn.model_selection import train_test_split

from project.figures.plots import hist, plot
from project.figures.rich import table
from project.funcs.common import load_data
from project.funcs.lda import lda
from project.funcs.pca import pca


def lab3(DATA: str):
    console = Console()

    X, y = load_data(DATA)

    _, PCA_data = pca(X, X.shape[1])
    hist(PCA_data, y, file_name="pca/histograms")

    _, LDA_data = lda(X, y, 1)
    hist(LDA_data, y, file_name="lda/histograms")

    # Classification using LDA

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    _, X_train_lda = lda(X_train, y_train, 1)
    _, X_val_lda = lda(X_val, y_val, 1)

    threshold = (
        X_train_lda[y_train == 0].mean() + X_train_lda[y_train == 1].mean()
    ) / 2.0

    y_pred = [0 if x >= threshold else 1 for x in X_val_lda.T[0]]

    table(
        console,
        "Mean of the projected class means",
        {
            "Threshold": [f"{threshold:.2f}"],
            "Error rate": [f"{np.sum(y_val != y_pred) / y_val.size * 100:.2f}%"],
        },
    )

    # Check if we can find a better threshold

    thresholds = np.linspace(X_train_lda.min(), X_train_lda.max(), 1000)
    empirical_error_rate = None
    empricial_threshold = None
    for threshold in thresholds:
        y_pred = [0 if x >= threshold else 1 for x in X_val_lda.T[0]]
        new_err = np.sum(y_val != y_pred) / y_val.size * 100

        if empirical_error_rate is None or new_err < empirical_error_rate:
            empirical_error_rate = new_err
            empricial_threshold = threshold

    table(
        console,
        "Brute force search",
        {
            "Threshold": [f"{empricial_threshold:.2f}"],
            "Error rate": [f"{empirical_error_rate:.2f}%"],
        },
    )

    error_rates_pca = []  # Error rates in percentage
    for i in range(1, X.shape[1] + 1):
        X_train_pca = pca(X_train, i)[1]

        _, X_train_lda = lda(X_train_pca, y_train, 1)
        _, X_val_lda = lda(X_val, y_val, 1)

        threshold = (
            X_train_lda[y_train == 0].mean() + X_train_lda[y_train == 1].mean()
        ) / 2.0

        # Predict the validation data
        y_pred = [0 if x >= threshold else 1 for x in X_val_lda.T[0]]

        # Calculate the error rate
        error_rates_pca.append(np.sum(y_val != y_pred) / y_val.size * 100)

    plot(
        dict({"Error rate": error_rates_pca}),
        range(1, X.shape[1] + 1),
        file_name="error_rate_pca",
        figsize=(8, 3),
    )
