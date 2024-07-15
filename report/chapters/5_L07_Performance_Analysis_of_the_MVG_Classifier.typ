#import "../funcs.typ": eqcolumns
#import "@preview/subpar:0.1.1"

#set page(
  header: align(right, text(fill: gray)[`Report for lab 7`]),
)

= Performance Analysis of the MVG Classifier

#eqcolumns(2)[
  == Comparing five applications

  When comparing the following five applications we notice how the cost of misclassification is reflected in the effective prior.

  #figure(
    align(center)[
      #set par(justify: false)
      #table(
        columns: 4,
        align: center + horizon,
        table.hline(stroke: 0.5pt),
        inset: 1em,
        table.cell(fill: luma(250), [#text(size: 1.2em, $bold(pi_T)$)]),
        table.cell(
          fill: luma(250),
          [#text(size: 1.2em, $bold(C_(f n))$)],
          inset: 1em,
        ),
        table.cell(
          fill: luma(250),
          [#text(size: 1.2em, $bold(C_(f p))$)],
          inset: 1em,
        ),
        table.cell(
          fill: luma(250),
          [#text(size: 1.2em, $bold(tilde(pi))$)],
          inset: 1em,
        ),
        table.hline(stroke: 0.5pt + gray),
        [0.5], [1.0], [1.0], [0.5],
        [0.9], [1.0], [1.0], [0.9],
        [0.1], [1.0], [1.0], [0.1],
        [0.5], [1.0], [9.0], [0.1],
        [0.5], [9.0], [1.0], [0.9],
        table.hline(stroke: 0.5pt),
      )
    ],
    caption: [Different priors and costs and their effect on the effective prior $tilde(pi)$],
  )

  We now consider the three unique effective priors and evaluate the best-performing model based on `minimum DCF`.

  / Application $bold(tilde(pi)) = 0.1$: The best model is the `Naïve Bayes` with a `minimum DCF` of `0.2570` and an `actual DCF` of `0.3022`. The best model was the one without `PCA` applied.

  / Application $bold(tilde(pi)) = 0.5$: The best model is the `Multivariate Gaussian` with a `minimum DCF` of `0.1302` and an `actual DCF` of `0.1399`. Again the best model was the one without `PCA` applied.

  / Application $bold(tilde(pi)) = 0.9$: The best model is again the `Multivariate Gaussian` one with a `minimum DCF` of `0.3423` and an `actual DCF` of `0.4001`. The best model was the one without `PCA` applied.

  === Considerations

  We notice that the binary Gaussian classifiers perform well with uniform prior and costs but become significantly worse when the effective prior is skewed. `PCA` did not improve the classification results in any of the applications.

  == Best PCA Setup for $bold(tilde(pi) = 0.1)$

  We notice something interesting, if we isolate our analysis to the application with effective prior equal to `0.1` and only consider the `PCA` setups, we see that the optimal number of dimensions for the `Multivariate Gaussian` model is `6`, with a `minimum DCF` of `0.3015` and an `actual DCF` of `0.3182`, but the other variants prefer working on lower dimensions, even though we saw before that all three perform best without `PCA`. When looking at the `Tied Covariance Gaussian` and the `Naïve Bayes` models we see, respectively, that the optimal number of dimensions is `4`, with a `minimum DCF` of `0.3721` and an `actual DCF` of `0.3879` and the optimal number of dimensions is just `1`, with a `minimum DCF` of `0.3730` and an `actual DCF` of `0.4049`.

  Looking at the `Bayes Error Plots` for the three models in the prior log odds range $(-4, +4)$, we see that the models' rankings are moderately consistent across the applications and are overall well calibrated. We notice that the `Tied Covariance` one is ever so slightly better calibrated than the others but worse overall while the `Multivariate Gaussian` one is the best-performing model overall.
]


#subpar.grid(
  figure(image("../imgs/naive_prior_log_odds.svg"), caption: [Naïve Bayes]),
  figure(
    image("../imgs/multivariate_prior_log_odds.svg"),
    caption: [Multivariate Gaussian],
  ),
  figure(
    image("../imgs/tied_prior_log_odds.svg"),
    caption: [Tied Covariance Gaussian],
  ),
  columns: 3,
  caption: [Bayes Error Plots for the three best PCA models with $tilde(pi) = 0.1$],
)
