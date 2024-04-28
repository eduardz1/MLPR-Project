import numpy as np
from sklearn.model_selection import train_test_split

from project.lda import lda
from project.logpdf import log_pdf
from project.pca import pca
from project.plots import (
    plot_error_rates,
    plot_gaussian_densities,
    plot_histograms,
    plot_scatter,
)

DATA = "data/trainData.txt"
CONF = {
    "lab2": False,
    "lab3": False,
    "lab4": True,
}


def lab2():
    dataset = np.loadtxt(DATA, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    plot_histograms(X, y, "features")
    plot_scatter(X, y)


def lab3():
    dataset = np.loadtxt(DATA, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    _, PCA_data = pca(X, X.shape[1])
    plot_histograms(PCA_data, y, "pca")

    _, LDA_data = lda(X, y, 1)
    plot_histograms(LDA_data, y, "lda")

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

    print(f"Threshold: {threshold:.2f}")
    print(f"Error rate: {np.sum(y_val != y_pred) / y_val.size * 100:.2f}%")

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

    print(
        f"Empirical error rate: {empirical_error_rate:.2f}%, found with threshold {empricial_threshold:.2f}"
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

    plot_error_rates(X, error_rates_pca)


def lab4():
    dataset = np.loadtxt(DATA, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    classes = np.unique(y)

    means = np.array([np.mean(X[y == c], axis=0) for c in classes])

    # If we want to analyze the data ina uni-variate way Covariance(X, X) = Var(X)
    vars = np.array([np.var(X[y == c], axis=0) for c in classes])

    plot_gaussian_densities(X, y, means, vars, log_pdf)

    print("ML estimates for the parameters are:")
    print(f"Means: {means}")
    print(f"Vars: {vars}")


def main():
    if CONF["lab2"]:
        lab2()

    if CONF["lab3"]:
        lab3()

    if CONF["lab4"]:
        lab4()


if __name__ == "__main__":
    main()
