"""
# PCA & LDA

Apply PCA and LDA to the project data. Start analyzing the effects of PCA on the
features. Plot the histogram of the projected features for the 6 PCA directions,
starting from the principal (largest variance). What do you observe? What are
the effects on the class distributions? Can you spot the different clusters
inside each class?

Apply LDA (1 dimensional, since we have just two classes), and compute the
histogram of the projected LDA samples. What do you observe? Do the classes
overlap? Compared to the histograms of the 6 features you computed in Laboratory
2, is LDA finding a good direction with little class overlap?

Try applying LDA as classifier. Divide the dataset in model training and
validation sets (you can reuse the previous function to split the dataset).
Apply LDA, select the orientation that results in the projected mean of class
True (label 1) being larger than the projected mean of class False (label 0),
and select the threshold as in the previous sections, i.e., as the average of
the projected class means. Compute the predictions on the validation data, and
the corresponding error rate.

Now try changing the value of the threshold. What do you observe? Can you find
values that improve the classification accuracy?

Finally, try pre-processing the features with PCA. Apply PCA (estimated on the
model training data only), and then classify the validation data with LDA.
Analyze the performance as a function of the number of PCA dimensions m. What do
you observe? Can you find values of m that improve the accuracy on the
validation set? Is PCA beneficial for the task when combined with the LDA
classifier?
"""

import numpy as np
from rich.console import Console

from project.figures.plots import hist, plot
from project.figures.rich import table
from project.funcs.base import load_data, split_db_2to1
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

    (X_train, y_train), (X_val, y_val) = split_db_2to1(X.T, y)

    _, X_train_lda = lda(X_train.T, y_train, 1)
    _, X_val_lda = lda(X_val.T, y_val, 1)

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
        X_train_pca = pca(X_train.T, i)[1]

        _, X_train_lda = lda(X_train_pca, y_train, 1)
        _, X_val_lda = lda(X_val.T, y_val, 1)

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
        xlabel="Number of PCA dimensions",
        ylabel="Error rate (%)",
    )
