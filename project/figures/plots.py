from multiprocessing import Pool, cpu_count
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from lxml import etree
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

# overestimates for 5 and 6 and underestimates for the other
# features when compared to the friedman-diaconis rule but it's
# preferable to have a consistent bin size across all features
BINS = "rice"
FIG_SIZE = (8, 6)  # Size of each figure in inches
IMG_FOLDER = "report/imgs"
POOL_SIZE = cpu_count() - 1


def _hist(args):
    X, y, i, kwargs = args

    plt.ioff()
    plt.figure(figsize=kwargs.get("figsize", FIG_SIZE))

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
    plt.savefig(f"{IMG_FOLDER}/hist/{kwargs.get('file_name')}_{i}.svg")
    plt.clf()


def hist(X, y, **kwargs):
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("Plotting histograms...   ", total=X.shape[1])

        with Pool(POOL_SIZE) as pool:
            for _ in pool.imap_unordered(
                _hist,
                [(X, y, i, kwargs) for i in range(X.shape[1])],
            ):
                progress.update(task_id, advance=1)


def _scatter(args):
    X, y, i, j, kwargs = args

    plt.ioff()
    plt.figure(figsize=kwargs.get("figsize", FIG_SIZE))

    # Plot first both classes so that x and y axis are the same size, then plot
    # the two classes separately to blend the svgs and better show the overlap
    # between the two classes, this is not possible at the moment directly in
    # matplotlib because there is no way to change the blending mode

    fakes = plt.scatter(
        X[y == 0][:, i],
        X[y == 0][:, j],
        alpha=0.6,
        color="red",
        rasterized=True,
    )
    genuines = plt.scatter(X[y == 1][:, i], X[y == 1][:, j])
    plt.tight_layout()
    genuines.remove()
    plt.savefig(
        f"{IMG_FOLDER}/scatter/single/fake_{i}_{j}.svg",
        transparent=True,
        dpi=300,
    )

    plt.scatter(
        X[y == 1][:, i],
        X[y == 1][:, j],
        alpha=0.6,
        color="blue",
        rasterized=True,
    )
    fakes.remove()
    plt.savefig(
        f"{IMG_FOLDER}/scatter/single/genuine_{i}_{j}.svg",
        transparent=True,
        dpi=300,
    )
    plt.clf()

    # Needed to better visualize difference between overlapping points, it's
    # faster to parse SVG files rather than using the .tostring() method on a
    # StringIO or BytesIO object
    blend_svgs(
        f"{IMG_FOLDER}/scatter/single/fake_{i}_{j}.svg",
        f"{IMG_FOLDER}/scatter/single/genuine_{i}_{j}.svg",
        f"{IMG_FOLDER}/scatter/{kwargs.get('file_name', 'overlay')}_{i}_{j}.svg",
    )


def scatter(X, y, **kwargs):
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Plotting scatter plots...", total=X.shape[1] ** 2 - X.shape[1]
        )

        with Pool(POOL_SIZE) as pool:
            for _ in pool.imap_unordered(
                _scatter,
                [
                    (X, y, i, j, kwargs)
                    for i in range(X.shape[1])
                    for j in range(X.shape[1])
                    if i != j
                ],
            ):
                progress.update(task, advance=1)


def plot(dict: dict[str, list], range: npt.ArrayLike, **kwargs) -> None:
    plt.ioff()
    plt.figure(figsize=kwargs.get("figsize", FIG_SIZE))

    for key, value in dict.items():
        plt.plot(range, value, label=key, marker=kwargs.get("marker", "o"))

    plt.xlabel(kwargs.get("xlabel", "x"))
    plt.ylabel(kwargs.get("ylabel", "y"))

    if "xscale" in kwargs:
        plt.xscale(kwargs.get("xscale", "log"), base=kwargs.get("base", 10))

    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{IMG_FOLDER}/{kwargs.get('file_name')}.svg")


def densities(
    X: npt.NDArray,
    y: npt.ArrayLike,
    means: npt.NDArray,
    vars: npt.NDArray,
    logpdf: Callable[[npt.NDArray, npt.NDArray, npt.NDArray], npt.NDArray],
):
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Plotting gaussian densities...", total=len(np.unique(y) * X.shape[1])
        )

        with Pool(POOL_SIZE) as pool:
            for _ in pool.imap_unordered(
                _densities,
                [
                    (X, y, means, vars, logpdf, i, j)
                    for i in np.unique(y)
                    for j in range(X.shape[1])
                ],
            ):
                progress.update(task, advance=1)


def _densities(args):
    X, y, means, vars, logpdf, i, j = args

    plt.ioff()
    plt.figure(figsize=FIG_SIZE)

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


def heatmap(X: npt.NDArray, cmap: str, file_name: str) -> None:
    plt.ioff()
    plt.figure(figsize=FIG_SIZE)

    sns.heatmap(
        X,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    plt.tight_layout()
    plt.savefig(f"{IMG_FOLDER}/heatmaps/{file_name}.svg")
    plt.clf()
