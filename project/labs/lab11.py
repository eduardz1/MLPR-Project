"""
# Calibration and Fusion

Consider the different classifiers that you trained in previous laboratories.
For each of the main methods (GMM, logistic regression, SVM — see Laboratory 10)
compute a calibration transformation for the scores of the best-performing
classifier you selected earlier. The calibration model should be trained using
the validation set that you employed in previous laboratories (i.e., the
validation split that you used to measure the systems performance). Apply a
K-fold approach to compute and evaluate the calibration transformation. You can
test different priors for training the logistic regression model, and evaluate
the performance of the calibration transformation in terms of actual DCF for the
target application (i.e., the training prior may be different than the target
application prior, but evaluation should be done for the target application).
For each model, select the best performing calibration transformation (i.e. the
one providing lowest actual DCF in the K-fold cross validation procedure for the
target application). Compute also the minimum DCF, and compare it to the actual
DCF, of the calibrated scores for the different systems. What do you observe?
Has calibration improved for the target application? What about different
applications (Bayes error plots)?

Compute a score-level fusion of the best-performing models. Again, you can try
different priors for training logistic regression, but you should select the
best model in terms of actual DCF computed for the target application. Compute
also the minimum DCF of the resulting model. How is the fusion performing? Is it
improving actual DCF with respect to single systems? Are the fused scores well
calibrated?

Choose the final model that will be used as “delivered” system, i.e. the final
system that will be used for application data. Justify your choice.
"""

from project.funcs.base import load_data, split_db_2to1


def lab11(DATA: str):
    X, y = load_data(DATA)

    (X_train, y_train), (X_val, y_val) = split_db_2to1(X.T, y)

    PRIOR = 0.1
