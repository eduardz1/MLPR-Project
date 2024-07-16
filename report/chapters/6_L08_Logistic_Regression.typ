#import "../funcs.typ": eqcolumns

#set page(
  header: align(right, text(fill: gray)[`Report for lab 8`]),
)

= Logistic Regression

#eqcolumns(2)[

  == Comparing the Effect of Regularization on the `DCF`

  We test the effect of the regularization coefficient $lambda$ on the `DCF` for the `Logistic Regression` model. We notice that incrementing the coefficient quickly makes the `actual DCF` increase. We can also notice that it does not affect the `minimum DCF`.

  #figure(
    caption: [Effects of the regularization coefficient $lambda$ on the `DCF` for the `Logistic Regression` model],
    image("../imgs/logreg/lambda_vs_dcf.svg", width: auto),
  )

  == Repeating the Analysis on a Smaller Dataset

  To better understand the effect of the regularization coefficient $lambda$ on the `DCF` we repeat the analysis on a smaller dataset. We notice that the `actual DCF` is more sensitive to the regularization coefficient $lambda$ than the `minimum DCF` either way but here, the latter benefits from a higher $lambda$. As far as the `actual DCF` is concerned, only one value, $10^(-2)$, stands out from the rest, providing an `actual DCF` value that is particularly close to the `minimum DCF`.

  We can clearly see how the model tends to overfit the data for $lambda < 10^(-2)$ and underfit for $lambda > 10^(-2)$. This information can be deduced by observing the regularized objective function that we minimize:

  $
    J(
      w, b
    ) = lambda / 2 ||w||^2 + 1 / n sum_(i=1)^n log(1 + e^(-z_i (w^T x_i + b))) \
    "with" z_i = cases(delim: "{", &1 "if" c_i = 1, -&1 "if" c_i = 0)
  $

  Here we see that a high value of $lambda$ will penalize the norm of the weights, leading to a "smoother" decision boundary, while a low value of $lambda$ will allow the model to fit the data more closely (and in the case we are looking at, overfit it for $lambda < 10^(-2)$).

  #figure(
    caption: [Effects of the regularization coefficient $lambda$ on the `DCF` for the `Logistic Regression` model on a smaller dataset],
    image("../imgs/logreg/lambda_vs_dcf_50.svg", width: auto),
  )

  == Repeating the Analysis on the Prior-Weighted Version of the Model

  Training with the prior-weighted version of the model does not lead to any relevant benefits. This could seem surprising, but it is not, given that the dataset is already very homogeneous.

  #figure(
    caption: [Effects of the regularization coefficient $lambda$ on the `DCF` for the prior-weighted version of the `Logistic Regression` model],
    image("../imgs/logreg/lambda_vs_dcf_prior.svg", width: auto),
  )
]

#pagebreak()

#eqcolumns(2)[
  == Repeating the Analysis on the Quadratic version of the Model

  We can repeat the analysis by expanding the feature space with the mapping

  $
    phi.alt (x) = mat(delim: "[", "vec"(x x^T); x)
  $

  Doing so leads to better results, which are expected given the nature of the dataset and, in particular the fact that features 5 and 6 are not linearly separable (see @features-5-6-scatter).

  In particular, for small values of $lambda$, the model is both especially well calibrated and performing well, when comparing the values of the `minimum DCF` to the previous models.

  #figure(
    caption: [Effects of the regularization coefficient $lambda$ on the `DCF` for the quadratic version of the `Logistic Regression` model (both in prior-weighted and non-prior-weighted versions)],
    image("../imgs/logreg/lambda_vs_dcf_quadratic.svg", width: auto),
  )

  Nonetheless, retraining the model with the prior-weighted version does lead to some marginal benefits in terms of both `minimum DCF` and `actual DCF` so we will consider this as the best-performing variant of the `Logistic Regression` model.

  == Repeating the Analysis with the Centered Dataset

  We notice very little difference by centering the dataset before training the model. This is because the dataset is already pretty much centered, as we noticed in the previous sections.

  #figure(
    caption: [Effects of the regularization coefficient $lambda$ on the `DCF` for the centered version of the `Logistic Regression` model],
    image("../imgs/logreg/lambda_vs_dcf_centered.svg", width: auto),
  )

  == Conclusions

  After analyzing the performance of different models for our application prior $0.1$, we conclude that the best performing one is the `Prior Weighted Quadratic Logistic Regression` with a $lambda = 0.0032$ with a `minimum DCF` of `0.2300` and an `actual DCF` of `0.2852`.

  The separation rule is quadratic (more specifically linear for the `expanded feature space` defined by the mapping $phi.alt$) and the distribution is assumed to have $tilde(pi) = 0.1$.

  This result is to be expected given in particular the nonlinear separability of features 5 and 6, as we saw in @features-5-6-scatter.
]
