import numpy as np

from project.funcs.logpdf import log_pdf
from project.funcs.plots import plot_gaussian_densities


def lab4(DATA: str):
    dataset = np.loadtxt(DATA, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    classes = np.unique(y)

    means = np.array([np.mean(X[y == c], axis=0) for c in classes])

    # If we want to analyze the data ina uni-variate way Covariance(X, X) = Var(X)
    vars = np.array([np.var(X[y == c], axis=0) for c in classes])

    plot_gaussian_densities(X, y, means, vars, log_pdf)

    print("ML estimates for the parameters are:")
    print(f"Means: {means}")
    print(f"Vars: {vars}")
