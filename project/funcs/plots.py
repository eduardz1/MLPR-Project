from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from lxml import etree

# overestimates for 5 and 6 and underestimates for the other
# features when compared to the friedman-diaconis rule but it's
# preferable to have a consistent bin size across all features
BINS = "rice"
FIG_SIZE = (8, 6)  # Size of each figure in inches
IMG_FOLDER = "report/imgs"


def config(figsize: tuple[int, int] = FIG_SIZE):
    plt.ioff()
    plt.figure(figsize=figsize)


def plot_histograms(X, y, data_type):
    config()

    for i in range(X.shape[1]):
        plt.hist(
            X.T[:, y == 0][i],
            bins=BINS,
            density=True,
            alpha=0.4,
            color="red",
        )
        plt.hist(
            X.T[:, y == 1][i],
            bins=BINS,
            density=True,
            alpha=0.4,
            color="blue",
        )
        plt.tight_layout()

        if data_type == "lda":
            plt.savefig(f"{IMG_FOLDER}/hist/lda/histograms_{i}.svg")
        elif data_type == "pca":
            plt.savefig(f"{IMG_FOLDER}/hist/pca/histograms_{i}.svg")
        else:
            plt.savefig(f"{IMG_FOLDER}/hist/histograms_{i}.svg")

        plt.clf()


def plot_scatter(X, y):
    config()

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

            plt.scatter(
                X[y == 0][:, i],
                X[y == 0][:, j],
                alpha=0.6,
                color="red",
                rasterized=True,
            )
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig(
                f"{IMG_FOLDER}/scatter/single/scatter_fake_{i}_{j}.svg",
                transparent=True,
                dpi=300,
            )
            plt.clf()

            plt.scatter(
                X[y == 1][:, i],
                X[y == 1][:, j],
                alpha=0.6,
                color="blue",
                rasterized=True,
            )
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig(
                f"{IMG_FOLDER}/scatter/single/scatter_genuine_{i}_{j}.svg",
                transparent=True,
                dpi=300,
            )
            plt.clf()

            blend_svgs(
                f"{IMG_FOLDER}/scatter/single/scatter_fake_{i}_{j}.svg",
                f"{IMG_FOLDER}/scatter/single/scatter_genuine_{i}_{j}.svg",
                f"{IMG_FOLDER}/scatter/overlay_{i}_{j}.svg",
            )


def plot_error_rates(X: npt.NDArray, y: npt.ArrayLike) -> None:
    config((8, 3))

    plt.plot(range(1, X.shape[1] + 1), y, marker="o")
    plt.tight_layout()
    plt.savefig(f"{IMG_FOLDER}/error_rate_pca.svg")
    plt.clf()


def plot_gaussian_densities(
    X: npt.NDArray,
    y: npt.ArrayLike,
    means: npt.NDArray,
    vars: npt.NDArray,
    logpdf: Callable[[npt.NDArray, npt.NDArray, npt.NDArray], npt.NDArray],
):
    config()

    for i in np.unique(y):
        for j in range(X.shape[1]):
            color = "red" if i == 0 else "blue"
            x = np.linspace(X[y == i][:, j].min(), X[y == i][:, j].max(), 100)

            plt.hist(
                X[y == i][:, j],
                bins=BINS,
                density=True,
                alpha=0.4,
                color=color,
            )

            plt.plot(
                x,
                np.exp(logpdf(x, means[i][j], vars[i][j])),
                color=color,
            )

            plt.tight_layout()
            plt.savefig(f"{IMG_FOLDER}/densities/density_{i}_{j}.svg")
            plt.clf()


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
