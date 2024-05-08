import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from project.funcs.common import load_data
from project.funcs.logpdf import log_pdf


def lab5(DATA: str):
    X, y = load_data(DATA)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    X_train = X_train.T
    X_val = X_val.T

    # Apply the MVG classifier
    accuracy, error_rate, llr = apply_mvg(X_train, X_val, y_train, y_val)

    print("\nMV Gaussian classifier")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Error rate: {error_rate:.2f}%")
    print(f"Log-likelihood ratio: {llr}")

    # Apply the tied Gaussian classifier
    accuracy, error_rate = apply_tied_gaussian(X_train, X_val, y_train, y_val)

    print("\nTied Gaussian classifier")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Error rate: {error_rate:.2f}%")

    # Apply the naive Gaussian classifier
    accuracy, error_rate = apply_naive_gaussian(X_train, X_val, y_train, y_val)

    print("\nNaive Gaussian classifier")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Error rate: {error_rate:.2f}%")

    # Display the covariance matrix of each class
    classes = np.unique(y_val)
    covariances = {k: np.cov(X_train[:, y_train == k], bias=True) for k in classes}

    print("\nCovariance matrices of each class:")
    print(covariances)

    # Plot covariances as heatmaps

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axs = plt.subplots(1, len(classes), figsize=(15, 5))
    for i, c in enumerate(classes):
        sns.heatmap(covariances[c], ax=axs[i], annot=True)
        axs[i].set_title(f"Class {c}")

    plt.show()


def apply_mvg(X_train, X_val, y_train, y_val) -> tuple[float, float, npt.ArrayLike]:
    """
    Returns:
        tuple[float, float]: Accuracy and error rate in percentage
    """

    classes = np.unique(y_val)

    S = np.zeros((len(classes), X_val.shape[1]))
    for i, c in enumerate(classes):
        mu = np.mean(X_train[:, y_train == c], axis=1, keepdims=True)
        C = np.cov(X_train[:, y_train == c], bias=True)

        log_pdf_val = log_pdf(X_val, mu, C)

        likelihood = np.exp(log_pdf_val)

        S[i, :] = likelihood

    llr = S[1] / S[0]

    # We assume uniform class priors
    SJoint = S * 0.5
    SMarginal = SJoint.sum(axis=0)
    SPost = SJoint / SMarginal

    y_pred = np.argmax(SPost, axis=0)

    accuracy = np.sum(y_val == y_pred) / y_val.size * 100
    error_rate = 100 - accuracy

    return accuracy, error_rate, llr


def apply_tied_gaussian(X_train, X_val, y_train, y_val) -> tuple[float, float]:
    """
    Returns:
        tuple[float, float]: Accuracy and error rate in percentage
    """

    classes = np.unique(y_val)
    weights = np.array([len(X_train[:, y_train == c]) for c in classes])
    Sw = np.average(
        [np.cov(X_train[:, y_train == c], bias=True) for c in classes],
        axis=0,
        weights=weights,
    )

    S = np.zeros((len(classes), X_val.shape[1]))
    for i, c in enumerate(classes):
        mu = np.mean(X_train[:, y_train == c], axis=1, keepdims=True)

        log_pdf_val = log_pdf(X_val, mu, Sw)

        likelihood = np.exp(log_pdf_val)

        S[i, :] = likelihood

    SJoint = S * 0.5
    SMarginal = SJoint.sum(axis=0)
    SPost = SJoint / SMarginal

    y_pred = np.argmax(SPost, axis=0)

    accuracy = np.sum(y_val == y_pred) / y_val.size * 100
    error_rate = 100 - accuracy

    return accuracy, error_rate


def apply_naive_gaussian(X_train, X_val, y_train, y_val) -> tuple[float, float]:
    """
    Returns:
        tuple[float, float]: Accuracy and error rate in percentage
    """

    classes = np.unique(y_val)

    S = np.zeros((len(classes), X_val.shape[1]))
    for i, c in enumerate(classes):
        mu = np.mean(X_train[:, y_train == c], axis=1, keepdims=True)
        C = np.cov(X_train[:, y_train == c], bias=True)

        # Diagonalize the covariance matrix
        C = np.diag(np.diag(C))

        log_pdf_val = log_pdf(X_val, mu, C)

        likelihood = np.exp(log_pdf_val)

        S[i, :] = likelihood

    SJoint = S * 0.5
    SMarginal = SJoint.sum(axis=0)
    SPost = SJoint / SMarginal

    y_pred = np.argmax(SPost, axis=0)

    accuracy = np.sum(y_val == y_pred) / y_val.size * 100
    error_rate = 100 - accuracy

    return accuracy, error_rate
