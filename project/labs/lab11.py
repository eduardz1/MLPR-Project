from project.funcs.base import load_data, split_db_2to1


def lab11(DATA: str):
    X, y = load_data(DATA)

    (X_train, y_train), (X_val, y_val) = split_db_2to1(X.T, y)

    PRIOR = 0.1
