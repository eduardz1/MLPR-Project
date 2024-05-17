import sklearn.metrics as skmetrics


def dcf(X, y_val, t, pi, Cf_n, Cf_p, normalize=False):
    """
    Compute the Detection Cost Function (DCF) for the given data and priors.

    Args:
        X (ArrayLike): The log-likelihood ratio (LLR) values.
        pi (float): The prior probability of a genuine sample.
        Cf_n (float): The cost of false negative.
        Cf_p (float): The cost of false positive.
        normalize (bool, optional): Whether to normalize the DCF value.
            Defaults to False.

    Returns:
        float: The DCF value.
    """

    y_pred = X > t

    cm = skmetrics.confusion_matrix(y_val, y_pred)

    P_fn = cm[1, 0] / cm[1].sum()
    P_fp = cm[0, 1] / cm[0].sum()

    return (pi * P_fn * Cf_n + (1 - pi) * P_fp * Cf_p) / (
        min(pi * Cf_n, (1 - pi) * Cf_p) if normalize else 1
    )
