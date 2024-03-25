import matplotlib.pyplot as plt
import numpy as np
from lxml import etree

DATA = "data/trainData.txt"
OPTIMIZE_SVGS = False
BINS = 30


def main():
    dataset = np.loadtxt(DATA, delimiter=",")

    # splits the dataset into features and labels
    X = dataset[:, :-1]
    y = dataset[:, -1]

    # Scale the features
    # X_scaled = preprocessing.normalize(X)

    # Plots histograms of the features

    plot_histograms(X, y)

    # # Plots of pairwise scatter plots of the features
    plot_scatter(X, y)

    # Apply PCA to the dataset
    PCA_eigvec, PCA_data = pca(X, X.shape[1])

    # Plots histograms of the PCA data
    plot_histograms_pca(PCA_data, y)

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


def pca(X, m=None):
    """Takes as input a data matrix X (with N samples and M features)
    and a number of features m to keep (with m <= M) and returns the eigenvectors
    and PCA data matrix. If m is not specified, all features are kept and will
    not be sorted.

    Returns:
        PCA_eigvec: [M x m] matrix with the eigenvectors of the covariance matrix
        PCA_data: [N x m] matrix with the PCA data
    """

    X = X - np.mean(X, axis=0)  # Center the data

    _, eigvec = np.linalg.eigh(np.cov(X.T))

    # Reverse so that the eigen vectors are sorted in
    # decreasing order and take the first m eigenvectors
    PCA_eigvec = eigvec[:, ::-1][:, :m]

    return PCA_eigvec, np.dot(X, PCA_eigvec)


def plot_histograms(X, y):
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
        plt.xlabel(f"Feature {i + 1}")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"report/imgs/hist/histograms_{i}.svg")
        plt.clf()


def plot_histograms_pca(PCA_data, y):
    for i in range(PCA_data.shape[1]):
        plt.hist(
            PCA_data.T[:, y == 0][i],
            bins=BINS,
            density=True,
            alpha=0.4,
            label="Fake",
            color="red",
        )
        plt.hist(
            PCA_data.T[:, y == 1][i],
            bins=BINS,
            density=True,
            alpha=0.4,
            label="Genuine",
            color="blue",
        )
        plt.xlabel(f"Feature {i + 1}")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"report/imgs/hist/pca/histograms_{i}.svg")
        plt.clf()


def plot_scatter(X, y):
    for i in range(X.shape[1]):
        for j in range(
            X.shape[1]
        ):  # generates duplicates but it's easier to work with in typst
            if i == j:
                continue

            # Extremely ugly but it normalizes dynamically tuples of features

            plt.scatter(X[y == 0][:, i], X[y == 0][:, j], alpha=0.4, color="red")
            plt.xlabel(f"Feature {i + 1}")
            plt.ylabel(f"Feature {j + 1}")
            xlim_0 = plt.gca().get_xlim()
            ylim_0 = plt.gca().get_ylim()
            plt.clf()

            plt.scatter(X[y == 1][:, i], X[y == 1][:, j], alpha=0.4, color="blue")
            plt.xlabel(f"Feature {i + 1}")
            plt.ylabel(f"Feature {j + 1}")
            xlim_1 = plt.gca().get_xlim()
            ylim_1 = plt.gca().get_ylim()
            plt.clf()

            xlim = [min(xlim_0[0], xlim_1[0]), max(xlim_0[1], xlim_1[1])]
            ylim = [min(ylim_0[0], ylim_1[0]), max(ylim_0[1], ylim_1[1])]

            plt.scatter(X[y == 0][:, i], X[y == 0][:, j], alpha=0.4, color="red")
            plt.xlabel(f"Feature {i + 1}")
            plt.ylabel(f"Feature {j + 1}")
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig(
                f"report/imgs/scatter/single/scatter_fake_{i}_{j}.svg", transparent=True
            )
            plt.clf()

            plt.scatter(X[y == 1][:, i], X[y == 1][:, j], alpha=0.4, color="blue")
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
