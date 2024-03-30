import matplotlib.pyplot as plt
import numpy as np
import scipy
from lxml import etree
from sklearn.model_selection import train_test_split

DATA = "data/trainData.txt"
OPTIMIZE_SVGS = False

# overestimates for 5 and 6 and underestimates for the other
# features when compared to the friedman-diaconis rule but it's
# preferable to have a consistent bin size across all features
BINS = "rice"

FIG_SIZE = (8, 6)  # Size of each figure in inches


def main():
    dataset = np.loadtxt(DATA, delimiter=",")

    # splits the dataset into features and labels
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    plt.figure(figsize=FIG_SIZE)

    # Plots histograms of the features
    # plot_histograms(X, y, "features")

    # # Plots of pairwise scatter plots of the features
    # plot_scatter(X, y)

    # Apply PCA to the dataset
    _, PCA_data = pca(X, X.shape[1])

    # Plots histograms of the PCA data
    # plot_histograms(PCA_data, y, "pca")

    # Apply LDA to the dataset
    _, LDA_data = lda(X, y, 1)

    # Plots histograms of the first direction of the LDA data
    # plot_histograms(LDA_data, y, "lda")

    # Classifications using LDA

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    # Fit the LDA model
    _, X_train_lda = lda(X_train, y_train, 1)
    _, X_val_lda = lda(X_val, y_val, 1)

    threshold = (
        X_train_lda[y_train == 0].mean() + X_train_lda[y_train == 1].mean()
    ) / 2.0

    # Predict the validation data
    y_pred = [0 if x >= threshold else 1 for x in X_val_lda.T[0]]

    print(f"Threshold: {threshold:.2f}")
    print(f"Error rate: {np.sum(y_val != y_pred) / y_val.size * 100:.2f}%")

    error_rates_pca = []  # Error rates in percentage
    for i in range(1, X.shape[1] + 1):
        X_train_pca = pca(X_train, i)[1]

        print(f"PCA with {i} features")
        print(X_train_pca.shape)

        _, X_train_lda = lda(X_train_pca, y_train, 1)
        _, X_val_lda = lda(X_val, y_val, 1)

        threshold = (
            X_train_lda[y_train == 0].mean() + X_train_lda[y_train == 1].mean()
        ) / 2.0

        # Predict the validation data
        y_pred = [0 if x >= threshold else 1 for x in X_val_lda.T[0]]

        # Calculate the error rate
        error_rates_pca.append(np.sum(y_val != y_pred) / y_val.size * 100)

    plt.plot(range(1, X.shape[1] + 1), error_rates_pca, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Error Rate (%)")
    plt.tight_layout()
    plt.savefig("report/imgs/error_rate_pca.svg")
    plt.clf()

    if OPTIMIZE_SVGS:
        import os
        import re

        from scour import scour

        # Optimize all SVGs under report/imgs
        # FIXME: broken rn
        for dirpath, _, files in os.walk("report/imgs"):
            for file in files:
                with open(os.path.join(dirpath, file), "r") as f:
                    if not re.search(r"\.svg$", file):
                        continue
                    svg = f.read()

                options = {
                    "enable_viewboxing": True,
                    "strip_ids": True,
                    "strip_comments": True,
                    "shorten_ids": True,
                    "indent_type": "none",
                }
                clean_svg = scour.scourString(svg, options)

                # Save the optimized SVG
                with open(os.path.join(dirpath, file), "w") as f:
                    f.write(clean_svg)


def split_db_2to1(D, L, seed=0):
    """Splits the dataset in a 2:1 ratio, taken from lab3.pdf

    Args:
        D (array_like): data matrix
        L (array_like): label vector
        seed (int, optional): seed for np.random. Defaults to 0.

    Returns:
        (DTR, LTR), (DVAL, LVAL): training and validation sets
    """


def pca(X, m):
    """Performs Principal Component Analysis on the data matrix X

    Args:
        X (array_like): [N x M] data matrix
        m (uint, optional): number of features to keep with m <= M

    Returns:
        The eigenvectors and PCA data matrix

        PCA_eigvec: [M x m] matrix with the eigenvectors of the covariance matrix
        PCA_data: [N x m] matrix with the PCA data
    """

    _, eigvec = np.linalg.eigh(np.cov(X.T))

    # Reverse so that the eigen vectors are sorted in
    # decreasing order and take the first m eigenvectors
    PCA_eigvec = eigvec[:, ::-1][:, :m]

    return PCA_eigvec, np.dot(X, PCA_eigvec)


def lda(X, y, m):
    """Performs Linear Discriminant Analysis on the data matrix X

    Args:
        X (array_like): [N x M] data matrix
        y (array_like): [N x 1] label vector
        m (uint, optional): number of features to keep with m <= M

    Returns:
        The eigenvectors and LDA data matrix

        LDA_eigvec: [M x m] matrix with the eigenvectors of the covariance matrix
        LDA_data: [N x m] matrix with the LDA data
    """

    means = np.array([np.mean(X[y == c], axis=0) for c in np.unique(y)])
    global_mean = np.mean(X, axis=0)
    unique_labels = np.unique(y)
    weights = np.array([len(X[y == c]) for c in unique_labels])

    # fmt: off
    # Compute the between-class covariance matrix
    Sb = np.average(
        [
            np.outer(means[c] - global_mean, means[c] - global_mean)
            for c in unique_labels
        ],
        axis=0,
        weights=weights
    )

    # Compute the within-class covariance matrix
    Sw = np.average(
        [   # ndmin=2 to handle the case where there is only one dimension
            np.array(np.cov(X[y == c].T), ndmin=2)
            for c in unique_labels
        ],
        axis=0,
        weights=weights
    )
    # fmt: on

    print(Sb.shape)
    print(Sw.shape)
    print([np.cov(X[y == c].T) for c in unique_labels])
    print(Sw)

    # Compute the eigenvectors of the generalized eigenvalue problem
    _, eigvec = scipy.linalg.eigh(Sb, Sw)

    # Reverse so that the eigen vectors are sorted in
    # decreasing order and take the first m eigenvectors
    LDA_eigvec = eigvec[:, ::-1][:, :m]

    return LDA_eigvec, np.dot(X, LDA_eigvec)


################################################################################
#                                    GRAPHICS                                  #
################################################################################


def plot_histograms(X, y, data_type):
    plt.figure(figsize=FIG_SIZE)
    for i in range(X.shape[1]):
        plt.hist(
            X.T[:, y == 0][i],
            bins=BINS,
            density=True,
            alpha=0.4,
            label="Fake",
            color="red",
        )
        plt.hist(
            X.T[:, y == 1][i],
            bins=BINS,
            density=True,
            alpha=0.4,
            label="Genuine",
            color="blue",
        )
        plt.legend()
        plt.tight_layout()

        if data_type == "lda":
            plt.xlabel(f"Direction {i + 1}")
            plt.savefig(f"report/imgs/hist/lda/histograms_{i}.svg")
        elif data_type == "pca":
            plt.xlabel(f"Principal Component {i + 1}")
            plt.savefig(f"report/imgs/hist/pca/histograms_{i}.svg")
        else:
            plt.xlabel(f"Feature {i + 1}")
            plt.savefig(f"report/imgs/hist/histograms_{i}.svg")

        plt.clf()


def plot_scatter(X, y):
    for i in range(X.shape[1]):
        for j in range(
            X.shape[1]
        ):  # generates duplicates but it's easier to work with in typst
            if i == j:
                continue

            # Extremely ugly but it normalizes dynamically tuples of features

            plt.scatter(X[y == 0][:, i], X[y == 0][:, j])
            xlim_0 = plt.gca().get_xlim()
            ylim_0 = plt.gca().get_ylim()
            plt.clf()

            plt.scatter(X[y == 1][:, i], X[y == 1][:, j])
            xlim_1 = plt.gca().get_xlim()
            ylim_1 = plt.gca().get_ylim()
            plt.clf()

            xlim = [min(xlim_0[0], xlim_1[0]), max(xlim_0[1], xlim_1[1])]
            ylim = [min(ylim_0[0], ylim_1[0]), max(ylim_0[1], ylim_1[1])]

            plt.scatter(X[y == 0][:, i], X[y == 0][:, j], alpha=0.6, color="red")
            plt.xlabel(f"Feature {i + 1}")
            plt.ylabel(f"Feature {j + 1}")
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig(
                f"report/imgs/scatter/single/scatter_fake_{i}_{j}.svg",
                transparent=True,
            )
            plt.clf()

            plt.scatter(X[y == 1][:, i], X[y == 1][:, j], alpha=0.6, color="blue")
            plt.xlabel(f"Feature {i + 1}")
            plt.ylabel(f"Feature {j + 1}")
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig(
                f"report/imgs/scatter/single/scatter_genuine_{i}_{j}.svg",
                transparent=True,
            )
            plt.clf()

            blend_svgs(
                f"report/imgs/scatter/single/scatter_fake_{i}_{j}.svg",
                f"report/imgs/scatter/single/scatter_genuine_{i}_{j}.svg",
                f"report/imgs/scatter/overlay_{i}_{j}.svg",
            )


def blend_svgs(svg1, svg2, path):
    # Parse the SVG files
    tree1 = etree.parse(svg1, parser=None)
    tree2 = etree.parse(svg2, parser=None)

    # Extract the root of each SVG file
    root1 = tree1.getroot()
    root2 = tree2.getroot()

    # Set the opacity of the elements in each SVG file to 0.5
    for elem in root1:
        elem.attrib["opacity"] = "0.5"
    for elem in root2:
        elem.attrib["opacity"] = "0.5"

    # Append the elements of the second SVG file to the first one
    for elem in root2:
        root1.append(elem)

    # Write the blended SVG to a new file
    with open(path, "wb") as f:
        f.write(etree.tostring(root1))


if __name__ == "__main__":
    main()
