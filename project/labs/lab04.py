import numpy as np
from rich.console import Console

from project.figures.plots import densities
from project.figures.rich import table
from project.funcs.common import load_data
from project.funcs.logpdf import log_pdf


def lab4(DATA: str):
    np.set_printoptions(precision=3, suppress=True)
    X, y = load_data(DATA)

    classes = np.unique(y)

    means = np.array([np.mean(X[y == c], axis=0) for c in classes])

    # If we want to analyze the data ina uni-variate way Covariance(X, X) = Var(X)
    vars = np.array([np.var(X[y == c], axis=0) for c in classes])

    densities(X, y, means, vars, log_pdf)

    table(
        Console(),
        "ML estimates for the parameters",
        {
            "means": [f"{means}"],
            "vars": [f"{vars}"],
        },
    )
