"""
Analyze the performance of the MVG classifier and its variants for different 
applications. Start considering five applications, given by (π1, Cf_n, Cf_p):

-   (0.5, 1.0, 1.0), i.e., uniform prior and costs
-   (0.9, 1.0, 1.0), i.e., the prior probability of a genuine sample is higher
    (in our application, most users are legit)
-   (0.1, 1.0, 1.0), i.e., the prior probability of a fake sample is higher 
    (in our application, most users are impostors)
-   (0.5, 1.0, 9.0), i.e., the prior is uniform (same probability of a legit and
    fake sample), but the cost of accepting a fake image is larger (granting 
    access to an impostor has a higher cost than labeling as impostor a legit
    user - we aim for strong security)
-   (0.5, 9.0, 1.0), i.e., the prior is uniform (same probability of a legit and
    fake sample), but the cost of rejecting a legit image is larger (granting 
    access to an impostor has a lower cost than labeling a legit user as
    impostor - we aim for ease of use for legit users)

Represent the applications in terms of effective prior. What do you obtain? 
Observe how the costs of mis-classifications are reflected in the prior: 
stronger security (higher false positive cost) corresponds to an equivalent 
lower prior probability of a legit user.

We now focus on the three applications, represented in terms of effective priors
(i.e., with costs of errors equal to 1) given by ˜π = 0.1, ˜π = 0.5 and 
˜π = 0.9, respectively.

For each application, compute the optimal Bayes decisions for the validation set
for the MVG models and its variants, with and without PCA (try different values 
of m). Compute DCF (actual) and minimum DCF for the different models. Compare 
the models in terms of minimum DCF. Which models perform best? Are relative 
performance results consistent for the different applications? Now consider also
actual DCFs. Are the models well calibrated (i.e., with a calibration loss in 
the range of few percents of the minimum DCF value) for the given applications? 
Are there models that are better calibrated than others for the considered
applications?

Consider now the PCA setup that gave the best results for the ˜π = 0.1 
configuration (this will be our main application). Compute the Bayes error plots
for the MVG, Tied and Naive Bayes Gaussian classifiers. Compare the minimum DCF 
of the three models for different applications, and, for each model, plot
minimum and actual DCF. Consider prior log odds in the range (−4, +4). What do
you observe? Are model rankings consistent across applications (minimum DCF)? 
Are models well-calibrated over the considered range?
"""

from project.classifiers.gaussian import Gaussian
from project.funcs.common import load_data


def lab7(DATA: str):
    X, y = load_data(DATA)

    _ = Gaussian(X, y)
