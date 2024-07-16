#import "../funcs.typ": eqcolumns

#set page(
  header: align(right, text(fill: gray)[`Report for lab 10`]),
)

= Gaussian Mixture Models

#eqcolumns(2)[
  == Full Covariance Matrix

  We start by looking for the best combination of positive and negative components for the `GMM` model with a full covariance matrix in the range $[1, 32]$, as we can see from the graph, the optimal number of components for the true class does not correspond to that of the false class. Indeed the number of components chosen has much more weight for the false class, while the true class performs generally well across the board.

  #figure(
    caption: [Grid search of the best combination of components for the positive and negative classes for the `GMM` model with a full covariance matrix (highlighted dots are the actual points tested)],
    image("../imgs/gmm/full.svg"),
  )

  #v(100pt)

  == Diagonal Covariance Matrix

  Looking at the diagonal covariance matrix variant, we notice that the behavior is similar but even more pronounced with just a small range of values where the false class performs well. This variant performs better than the full covariance matrix one and that is likely due to the fact that the diagonalization transformation reduces overfitting. We saw in @correlation-matrices that the features are largely uncorrelated.

  #figure(
    caption: [Grid search of the best combination of components for the positive and negative classes for the `GMM` model with a diagonal covariance matrix (highlighted dots are the actual points tested)],
    image("../imgs/gmm/diagonal.svg"),
  )
]

#v(-50pt)

== Best GMM Combination

#[
  #set par(justify: false)
  #table(
    columns: 5,
    align: center + horizon,
    table.hline(stroke: 0.5pt),
    inset: 1em,
    table.cell(fill: luma(250), [*Covariance Type*]),
    table.cell(fill: luma(250), [*Positive Components*], inset: 1em),
    table.cell(fill: luma(250), [*Negative Components*], inset: 1em),
    table.cell(fill: luma(250), [*Min DCF*], inset: 1em),
    table.cell(fill: luma(250), [*Act DCF*], inset: 1em),
    table.hline(stroke: 0.5pt + gray),
    [Diagonal], [32], [8], [0.1312], [0.1517],
    table.hline(stroke: 0.5pt),
  )]

Perhaps surprisingly, increasing the number of components for the `True` class is always beneficial to the task, much less surprising the fact that a minimum number of negative components is needed for good classification, given that we have seen in @gaussian-models[Lab 4 Section] that the false class looks like a combination of multiple Gaussians.

The diagonal covariance model performs better than the full one and I believe that to be thanks to the fact that we are using multiple Gaussian distributions to model the data. We have seen in @filter-last-two that the `Naïve Bayes` model already performs better than the `Multivariate Gaussian` model without the last two features and these last two features are the ones that most benefit from the `GMM` model.

== Best Models Compared for Different Applications

#[
  #set par(justify: false)
  #figure(
    table(
      columns: 4,
      align: center + top,
      table.hline(stroke: 0.5pt),
      inset: 1em,
      table.cell(fill: luma(250), [*DCF*], align: (center + horizon)),
      table.cell(
        fill: luma(250),
        [*GMM* $ N^o_"positive" &= 32\ N^o_"negative" &= 8 $ diagonal covariance],
        inset: 1em,
      ),
      table.cell(
        fill: luma(250),
        align(top)[*Logistic Regression* $ lambda approx 0.0032\ "prior weighted"\ "quadratic" $],
        inset: 1em,
      ),
      table.cell(
        fill: luma(250),
        [*SVM* $ "RBF kernel" $ $
            C &approx 32\ gamma &approx 0.135 \ xi &= 1
          $],
        inset: 1em,
      ),
      table.hline(stroke: 0.5pt + gray),
      [minimum], [0.1312], [0.2300], [0.1755],
      [actual], [0.1517], [0.2852], [0.4216],
      table.hline(stroke: 0.5pt)
    ),
    caption: [Scores of the best models for the application with effective prior $tilde(pi) = 0.1$],
  )]

By comparing our best models so far on a wide range of applications we can see that the `GMM` model is by far the best-performing one and is also the best calibrated. The `SVM` model, in particular, is very badly calibrated (expected outcome given the non-probabilistic nature of the classifier) but it has the potential to outperform the `Logistic Regression` one.

We Expected the `GMM` model to perform the best compared to the others given that our Gaussian assumptions hold very well for the dataset, or more precisely, hold well when considering the distributions as a combination of multiple Gaussian distributions.

There are no models here that are harmful for the specified applications.

#figure(
  image("../imgs/best_models_dcf.svg"),
  caption: [Comparison of the Bayes error plots for the best models for the application with effective prior $tilde(pi) = 0.1$],
)
