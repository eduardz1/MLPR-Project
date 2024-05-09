import numpy as np
from rich.align import Align
from rich.console import Console
from rich.table import Table

from project.funcs.common import load_data
from project.funcs.logpdf import log_pdf
from project.funcs.plots import plot_gaussian_densities


def lab4(DATA: str):
    np.set_printoptions(precision=3, suppress=True)
    X, y = load_data(DATA)

    classes = np.unique(y)

    means = np.array([np.mean(X[y == c], axis=0) for c in classes])

    # If we want to analyze the data ina uni-variate way Covariance(X, X) = Var(X)
    vars = np.array([np.var(X[y == c], axis=0) for c in classes])

    plot_gaussian_densities(X, y, means, vars, log_pdf)

    console = Console()
    table = Table(title="ML estimates for the parameters")

    table.add_column("means", justify="center")
    table.add_column("vars", justify="center")

    table.add_row(
        f"{means}",
        f"{vars}",
    )
    console.print(Align.center(table), new_line_start=True)
