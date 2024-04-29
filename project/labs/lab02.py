import numpy as np

from project.funcs.plots import plot_histograms, plot_scatter


def lab2(DATA: str):
    dataset = np.loadtxt(DATA, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    plot_histograms(X, y, "features")
    plot_scatter(X, y)
