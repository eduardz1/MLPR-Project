"""
# Probability densities and ML estimates

Try fitting uni-variate Gaussian models to the different features of the 
different classes of the project dataset. For each class, for each component of
the feature vector of that class, compute the ML estimate for the parameters of 
a 1D Gaussian distribution. Plot the distribution density (remember that you 
have to exponentiate the log-density) on top of the normalized histogram (set 
density=True when creating the histogram, see Laboratory 2). What do you 
observe? Are there features for which the Gaussian densities provide a good fit?
Are there features for which the Gaussian model seems significantly less 
accurate?

## Note:
for this part of the project, since we are still performing some preliminary,
qualitative analysis, you can compute the ML estimates and the plots either on 
the whole training set. In the following labs we will employ the densities for
classification, and we will need to perform model selection, therefore we will 
re-compute ML estimates on the model training portion of the dataset only 
(see Laboratory 3).
"""

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
