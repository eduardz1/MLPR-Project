import numpy as np


def vcol(vec):
    return vec.reshape(-1, 1)


def vrow(vec):
    return vec.reshape(1, -1)


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the data from the given path

    Args:
        path (str): Path to the data file

    Returns:
        tuple[np.ndarray, np.ndarray]: Features and labels
    """
    dataset = np.loadtxt(path, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    return X, y
