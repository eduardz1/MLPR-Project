from project.figures.plots import hist, scatter
from project.funcs.common import load_data


def lab2(DATA: str):
    X, y = load_data(DATA)

    hist(X, y, file_name="histograms")

    scatter(X, y, file_name="overlay")
