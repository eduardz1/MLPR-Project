from project.funcs.common import load_data
from project.funcs.plots import plot_histograms, plot_scatter


def lab2(DATA: str):
    X, y = load_data(DATA)

    plot_histograms(X, y, "features")
    plot_scatter(X, y)
