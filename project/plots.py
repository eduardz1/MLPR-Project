import matplotlib.pyplot as plt
from lxml import etree

# overestimates for 5 and 6 and underestimates for the other
# features when compared to the friedman-diaconis rule but it's
# preferable to have a consistent bin size across all features
BINS = "rice"
FIG_SIZE = (8, 6)  # Size of each figure in inches
IMG_FOLDER = "report/imgs"


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
            plt.savefig(f"{IMG_FOLDER}/hist/lda/histograms_{i}.svg")
        elif data_type == "pca":
            plt.xlabel(f"Principal Component {i + 1}")
            plt.savefig(f"{IMG_FOLDER}/hist/pca/histograms_{i}.svg")
        else:
            plt.xlabel(f"Feature {i + 1}")
            plt.savefig(f"{IMG_FOLDER}/hist/histograms_{i}.svg")

        plt.clf()


def plot_scatter(X, y):
    plt.figure(figsize=FIG_SIZE)

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
                f"{IMG_FOLDER}/scatter/single/scatter_fake_{i}_{j}.svg",
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
                f"{IMG_FOLDER}/scatter/single/scatter_genuine_{i}_{j}.svg",
                transparent=True,
            )
            plt.clf()

            blend_svgs(
                f"{IMG_FOLDER}/scatter/single/scatter_fake_{i}_{j}.svg",
                f"{IMG_FOLDER}/scatter/single/scatter_genuine_{i}_{j}.svg",
                f"{IMG_FOLDER}/scatter/overlay_{i}_{j}.svg",
            )


def plot_plot(X, y):
    plt.figure(figsize=FIG_SIZE)

    plt.plot(range(1, X.shape[1] + 1), y, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Error Rate (%)")
    plt.tight_layout()
    plt.savefig(f"{IMG_FOLDER}/error_rate_pca.svg")
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
