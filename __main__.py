import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# from sklearn import preprocessing

DATA = "data/trainData.txt"


def main():
    dataset = np.loadtxt(DATA, delimiter=",")

    # splits the dataset into features and labels
    X = dataset[:, :-1]
    y = dataset[:, -1]

    # Scale the features
    # X_scaled = preprocessing.normalize(X)

    # Plots histograms of the features
    plot_histograms(X, y)

    # Plots of pairwise scatter plots of the features
    plot_scatter(X, y)

    # Apply PCA to the dataset
    PCA_eigenvec, PCA_data = pca(X, 6)


def pca(X, m):
    """Takes as input a data matrix X (with N samples and M features)
    and a number of features m to keep (with m <= M) and returns the eigenvectors
    and PCA data matrix

    Returns:
        PCA_eigvec: [M x m] matrix with the eigenvectors of the covariance matrix
        PCA_data: [N x m] matrix with the PCA data
    """

    X = X - np.mean(X, axis=0)  # Center the data

    eigval, eigvec = np.linalg.eig(np.cov(X.T))

    # Sort eigenvalues and eigenvectors in descending order

    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    PCA_eigvec = eigvec[:, :m]
    PCA_data = np.dot(X, PCA_eigvec)

    return PCA_eigvec, PCA_data


def plot_histograms(X, y):
    for i in range(X.shape[1]):
        plt.hist(
            X.T[:, y == 0][i],
            bins=10,
            density=True,
            alpha=0.4,
            label="Fake",
            color="red",
        )
        plt.hist(
            X.T[:, y == 1][i],
            bins=10,
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


def plot_scatter(X, y):
    for i in range(X.shape[1]):
        for j in range(
            X.shape[1]
        ):  # generates duplicates but it's easier to work with in typst
            if i == j:
                continue

            # Extremely ugly but it normalizes dinamically tuples of features

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
            plt.savefig(f"report/imgs/scatter/single/scatter_fake_{i}_{j}.png")
            plt.clf()

            plt.scatter(X[y == 1][:, i], X[y == 1][:, j], alpha=0.4, color="blue")
            plt.xlabel(f"Feature {i + 1}")
            plt.ylabel(f"Feature {j + 1}")
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig(f"report/imgs/scatter/single/scatter_genuine_{i}_{j}.png")
            plt.clf()

            img1 = Image.open(f"report/imgs/scatter/single/scatter_fake_{i}_{j}.png")
            img2 = Image.open(f"report/imgs/scatter/single/scatter_genuine_{i}_{j}.png")

            new_img = Image.blend(img1, img2, alpha=0.5)
            new_img.save(f"report/imgs/scatter/overlay_{i}_{j}.png", "PNG")


if __name__ == "__main__":
    main()
