#import "@preview/oxifmt:0.2.0": strfmt
#import "template.typ": *
#import "funcs.typ": *
#import "@preview/subpar:0.1.1"

#show: template

#heading(level: 2, numbering: none)[Introduction]

#eqcolumns(2)[
  The task consists of a binary classification problem, the goal is to perform fingerprint spoofing detection (i.e. to distinguish between real and fake fingerprints).
  The dataset consists of 6 features. In this first part, we will analyze some statistics of the dataset and the correlation between the features.
]

#let cells = ()
#for i in range(0, 6) {
  for j in range(0, 6) {
    if (i != j) {
      cells.push(
        table.cell(
          inset: 0em,
          x: i + 1,
          y: j + 1,
          image(strfmt("imgs/scatter/overlay_{}_{}.svg", i, j)),
        ),
      )
    } else {
      cells.push(
        table.cell(
          inset: 0em,
          x: i + 1,
          y: j + 1,
          image(strfmt("imgs/hist/histograms_{}.svg", i)),
        ),
      )
    }
  }
}

#figure(
  caption: [Summary of the dataset features plotted against each other, number corresponds to the feature number],
  [
    #set text(size: 0.7em)
    #table(
      align: center + horizon,
      stroke: none,
      columns: 7,
      rows: 3,
      [],
      [1],
      [2],
      [3],
      [4],
      [5],
      [6],
      table.cell(x: 0, y: 1, rotate(-90deg, reflow: true)[1]),
      table.cell(x: 0, y: 2, rotate(-90deg, reflow: true)[2]),
      table.cell(x: 0, y: 3, rotate(-90deg, reflow: true)[3]),
      table.cell(x: 0, y: 4, rotate(-90deg, reflow: true)[4]),
      table.cell(x: 0, y: 5, rotate(-90deg, reflow: true)[5]),
      table.cell(x: 0, y: 6, rotate(-90deg, reflow: true)[6]),
      ..cells,
    )
  ],
  kind: image,
) <scatter>

#v(1fr)

#set page(
  header: align(right, text(fill: gray)[`Report for lab 2`]),
)

= Features Compared
== Features 1 and 2 <features-1-2>
#grid(
  columns: 2,
  figure(image("imgs/hist/histograms_0.svg"), caption: [Feature 1]),
  figure(image("imgs/hist/histograms_1.svg"), caption: [Feature 2]),
)
#eqcolumns(2)[
  When looking at the first feature we can observe that the classes overlap almost completely. The `Genuine` label has a higher variance than the `Fake` class but the mean is similar. Both classes exhibit one mode in the histogram but the `Fake` class has a higher peak. Looking at the second feature we can notice the opposite behavior. The `Fake` class has a higher variance than the `Genuine` class but the mean is similar. Both classes exhibit one mode in the histogram but
  the `Genuine` class has a higher peak. Again, the classes overlap almost completely.
]

#line(length: 100%, stroke: 0.5pt + gray)

== Features 3 and 4
#grid(
  columns: 2,
  figure(image("imgs/hist/histograms_2.svg"), caption: [Feature 3]),
  figure(image("imgs/hist/histograms_3.svg"), caption: [Feature 4]),
)
#eqcolumns(2)[
  Looking at the plot for the third class we can notice that the two features are much more distinct, they overlap slightly in 0. The `Genuine` class has a peak in -1 while the `Fake` class has a peak in 1. They both have similar variance but the means differ. One mode for each class is evident from the histogram.
  The fourth feature shows similar characteristics to the third feature.
]

#pagebreak()

== Features 5 and 6
#grid(
  columns: 2,
  figure(image("imgs/hist/histograms_4.svg"), caption: [Feature 5]),
  figure(image("imgs/hist/histograms_5.svg"), caption: [Feature 6]),
)
#eqcolumns(2)[
  The fifth feature also shows a good distinction between the two classes with an overlap at the edges of the `Fake` class distribution. They exhibit similar variance but with a lower mean for the `Genuine` class. The `Fake` class peaks in 0 while the `Genuine` has two modes and peaks in -1 and 1.
  The last feature shows similar characteristics to the fifth feature.
]

#line(length: 100%, stroke: 0.5pt + gray)

#figure(image("imgs/scatter/overlay_4_5.svg"), caption: [Features 5 and 6 Scatter Plot])

Looking at the scatter plot we see that there are four distinct clusters for each of the labels, they overlap slightly at the edges of each cluster.

#set page(
  header: align(right, text(fill: gray)[`Report for lab 3`]),
)

= PCA & LDA

#columns(2)[
  == Principal Component Analysis

  #figure(
    caption: [Principal components computed by PCA, ordered from top to bottom, right to left],
    grid(
      columns: 2,
      rows: 3,
      image("imgs/hist/pca/histograms_0.svg"), image("imgs/hist/pca/histograms_1.svg"),
      image("imgs/hist/pca/histograms_2.svg"), image("imgs/hist/pca/histograms_3.svg"),
      image("imgs/hist/pca/histograms_4.svg"), image("imgs/hist/pca/histograms_5.svg"),
    ),
  )

  Looking at the principal components of the dataset we can see that only one results in a clear separation between the two classes and it seems to separate the two classes better than any other feature taken individually. The rotation makes it harder to see the clusters, if we were to plot the data in 6D we would see the same original clusters.

  == Linear Discriminant Analysis

  #figure(caption: [First LDA direction], image("imgs/hist/lda/histograms_0.svg"))

  We see that compared to the first principal the classes are mirrored but the separation is similar between the two methods.

  === Applying LDA as a classifier <lda-classifier>

  We now try to apply LDA as a classifier, we start by splitting the dataset in a training and validation set, then we fit the model on the training and evaluation set, then we calculate the optimal threshold for the classifier and finally, we evaluate the model on the validation set.

  ```python
  # Split the dataset
  (X_train, y_train), (X_val, y_val) = split_db_2to1(
    X.T, y
  )

  # Fit the LDA model
  _, X_train_lda = lda(X_train.T, y_train, 1)
  _, X_val_lda = lda(X_val.T, y_val, 1)

  threshold = (
      X_train_lda[y_train == 0].mean() +
      X_train_lda[y_train == 1].mean()
  ) / 2.0

  # Predict the validation data
  y_pred = [
    0
    if x >= threshold
    else 1
    for x in X_val_lda.T[0]
  ]
  ```

  ```
  Threshold: -0.02
  Error rate: 9.35%
  ```

  Empirically we can find that threshold `0.15` gives a slightly better error rate of `9.10%`.

  === Pre-processing the Data with PCA

  #figure(
    caption: [Error rates in percentage as a function of the number of PCA directions],
    image("imgs/error_rate_pca.svg"),
  )

  As we can see from the graph, pre-processing the data with PCA does not improve the classification results. With the number of dimensions set to 2 or 3, the accuracy even decreases slightly.
]

#pagebreak()

#set page(
  header: align(right, text(fill: gray)[`Report for lab 4`]),
)

= ML Estimates & Probability Densities

== Gaussian Models <gaussian-models>
#figure(
  caption: [Uni-variate Gaussian models with a good fit to the data],
  table(
    columns: 4,
    stroke: none,
    [1],
    [2],
    [3],
    [4],
    table.cell(inset: 0em, image("imgs/densities/density_0_0.svg")),
    table.cell(inset: 0em, image("imgs/densities/density_0_1.svg")),
    table.cell(inset: 0em, image("imgs/densities/density_0_2.svg")),
    table.cell(inset: 0em, image("imgs/densities/density_0_3.svg")),
    table.cell(inset: 0em, image("imgs/densities/density_1_0.svg")),
    table.cell(inset: 0em, image("imgs/densities/density_1_1.svg")),
    table.cell(inset: 0em, image("imgs/densities/density_1_2.svg")),
    table.cell(inset: 0em, image("imgs/densities/density_1_3.svg")),
  ),
)
#columns(2)[
  It is noticeable that features 1, 2, 3, and 4 fit well to a uni-variate Gaussian model, both for the `Genuine` and `Fake` classes.

  #figure(
    caption: [Uni-variate Gaussian models with bad fit to the data],
    table(
      columns: 2,
      stroke: none,
      [4],
      [5],
      table.cell(inset: 0em, image("imgs/densities/density_0_4.svg")),
      table.cell(inset: 0em, image("imgs/densities/density_0_5.svg")),
      table.cell(inset: 0em, image("imgs/densities/density_1_4.svg")),
      table.cell(inset: 0em, image("imgs/densities/density_1_5.svg")),
    ),
  )

  When looking at features 5 and 6, on the other hand, we can see that the uni-variate Gaussian model does not fit the data well. The `Genuine` class in particular has a bimodal distribution for both features so it results in a particularly bad fit.

  == Maximum Likelihood Estimates

  $
    mu_("ML") = 1 / N sum_(i=1)^N x_i, space space sigma^2_("ML") = 1 / N sum_(i=1)^N (x_i - mu_("ML"))^2
  $

  The ML estimates for the parameters of a `Uni-variate Gaussian` model correspond to the dataset mean and variance for each feature.

  The following table summarizes the ML estimates for the dataset features.

  // $
  //   mu_("ML") = mat(
  //     0.003, 0.019, -0.681, 0.671, 0.028, -0.006;
  //     0.001, -0.009, 0.665, -0.664, -0.042, 0.024; delim: "["
  //   )
  // $

  #table(
    columns: (auto, 1fr, 1fr),
    align: center + horizon,
    table.hline(stroke: 0.5pt),
    table.cell(fill: luma(250), colspan: 3, [#text(size: 1.2em, $bold(mu_("ML"))$)], inset: 1em),
    table.hline(stroke: 0.5pt + gray),
    [*\#*],
    [*Fake*],
    [*Genuine*],
    [1],
    [#text(size: 0.8em, $space space 2.87744301 dot 10^(-3)$)],
    [#text(size: 0.8em, $space space 5.44547838 dot 10^(-4)$)],
    [2],
    [#text(size: 0.8em, $space space 1.86931579 dot 10^(-2)$)],
    [#text(size: 0.8em, $-8.52437392 dot 10^(-3)$)],
    [3],
    [#text(size: 0.8em, $-6.80940159 dot 10^(-1)$)],
    [#text(size: 0.8em, $space space 6.65237846 dot 10^(-1)$)],
    [4],
    [#text(size: 0.8em, $space space 6.70836195 dot 10^(-1)$)],
    [#text(size: 0.8em, $-6.64195349 dot 10^(-1)$)],
    [5],
    [#text(size: 0.8em, $space space 2.79569669 dot 10^(-2)$)],
    [#text(size: 0.8em, $-4.17251858 dot 10^(-2)$)],
    [6],
    [#text(size: 0.8em, $-5.82740035 dot 10^(-3)$)],
    [#text(size: 0.8em, $space space 2.39384879 dot 10^(-2)$)],
    table.hline(stroke: 0.5pt),
  )

  #table(
    columns: (auto, 1fr, 1fr),
    align: center + horizon,
    table.hline(stroke: 0.5pt),
    table.cell(fill: luma(250), colspan: 3, [#text(size: 1.2em, $bold(sigma^2_("ML"))$)], inset: 1em),
    table.hline(stroke: 0.5pt + gray),
    [*\#*],
    [*Fake*],
    [*Genuine*],
    [1],
    [#text(size: 0.8em, $0.56958105$)],
    [#text(size: 0.8em, $1.43023345$)],
    [2],
    [#text(size: 0.8em, $1.42086571$)],
    [#text(size: 0.8em, $0.57827792$)],
    [3],
    [#text(size: 0.8em, $0.54997702$)],
    [#text(size: 0.8em, $0.54890260$)],
    [4],
    [#text(size: 0.8em, $0.53604266$)],
    [#text(size: 0.8em, $0.57827792$)],
    [5],
    [#text(size: 0.8em, $0.68007360$)],
    [#text(size: 0.8em, $0.55334275$)],
    [6],
    [#text(size: 0.8em, $0.70503844$)],
    [#text(size: 0.8em, $1.28702609$)],
    table.hline(stroke: 0.5pt),
  )
]

#pagebreak(weak: true)

#set page(
  header: align(right, text(fill: gray)[`Report for lab 5`]),
)

= Generative Models for Classification

#align(center)[
  #figure(
    [
      #set par(justify: false)
      #table(
        columns: 4,
        align: center + horizon,
        table.hline(stroke: 0.5pt),
        inset: 1em,
        table.cell(fill: luma(250), []),
        table.cell(fill: luma(250), [*Multivariate*], inset: 1em),
        table.cell(fill: luma(250), [*Tied Covariance*], inset: 1em),
        table.cell(fill: luma(250), [*Naïve Bayes*], inset: 1em),
        table.hline(stroke: 0.5pt + gray),
        [*Accuracy*],
        [93.00%],
        [90.70%],
        [92.80%],
        [*Error Rate*],
        [7.00%],
        [9.30%],
        [7.20%],
        table.hline(stroke: 0.5pt),
      )
    ],
    caption: [Different Gaussian classifiers perfomance compared],
  ) <all-features>
]

\

The table above summarizes the various results, showing that the `Multi Variate Gaussian` model performs the best with an accuracy of `93%` and an error rate of `7%`. The `Tied Covariance Gaussian` model is the one that performs the worst. All the models perform better than `LDA` (see @lda-classifier[Lab 3 Section]).

\

#eqcolumns(2)[
  == Covariance and Correlation Matrices for Fake and Genuine Classes

  To better visualize the correlations we can view them as heatmaps.

  #figure(
    caption: [Covariance matrices for the `Fake` and `Genuine` classes],
    grid(
      columns: 2,
      inset: (right: -5pt, left: -5pt),
      image("imgs/heatmaps/covariance_fake.svg"), image("imgs/heatmaps/covariance_genuine.svg"),
    ),
  )

  We notice the covariance values are very small compared to the variances. To better visualize the strength of the variances with respect to the covariances we can compute the correlation matrices for the two classes.

  #figure(
    caption: [Correlation matrices for the `Fake` and `Genuine` classes],
    grid(
      columns: 2,
      inset: (right: -5pt, left: -5pt),
      image("imgs/heatmaps/correlation_fake.svg"), image("imgs/heatmaps/correlation_genuine.svg"),
    ),
  )

  From the correlation matrices, we can see that the features are weakly correlated with each other. Given the Naïve Bayes assumption of indipendence, $p(E | c) = product_(i = 1)^n p (x_i | c)$ where $E = (x_1, x_2, dots, x_n)$, it's not surprising that the model performs well. (Note that indipendence is not really needed for the model to perform well, #cite(<Zhang2004TheOO>, form: "prose") explores the optimality of the Naïve Bayes classifier even in in the presence correlation between the features).

  == Filtering out the Last Two Features

  We noticed in @gaussian-models[Lab 4 Section] that the last two features do not fit well with the Gaussian assumption. When repeating the classification tasks without the last two features, we notice that the accuracy decreses slightly across the board. This result, pheraphs unexpected, implies that there is still valuable information to be extracted from these features.

  #figure(
    align(center)[
      #set par(justify: false)
      #table(
        columns: 3,
        align: center + horizon,
        table.hline(stroke: 0.5pt),
        inset: 1em,
        table.cell(fill: luma(250), []),
        table.cell(fill: luma(250), [*Accuracy*], inset: 1em),
        table.cell(fill: luma(250), [*Error Rate*], inset: 1em),
        table.hline(stroke: 0.5pt + gray),
        [*Multivariate*],[92.05%],[7.95%],
        [*Tied Covariance*],[90.50%],[9.50%],
        [*Naïve Bayes*],[92.35%],[7.65%],
        table.hline(stroke: 0.5pt),
      )
    ],
    caption: [Classification results without the last two features],
  ) <filter-last-two>

  == First Two Features

  When we apply the Multivariate and Tied Covariance Gaussian classifiers on only the first two features, we notice that the accuracy decreases drastically. This is to be expected, as we have seen in @features-1-2[Lab 2 Section] and particularly in @scatter, the two features don't discriminate well between the two classes and the combination of the two doesn't help either.

  #figure(
    align(center)[
      #set par(justify: false)
      #table(
        columns: 3,
        align: center + horizon,
        table.hline(stroke: 0.5pt),
        inset: 1em,
        table.cell(fill: luma(250), []),
        table.cell(fill: luma(250), [*Accuracy*], inset: 1em),
        table.cell(fill: luma(250), [*Error Rate*], inset: 1em),
        table.hline(stroke: 0.5pt + gray),
        [*Multivariate*], [63.50%], [36.50%],
        [*Tied Covariance*], [50.55%], [49.45%],
        table.hline(stroke: 0.5pt),
      )
    ],
    caption: [Classification results for the first two features],
  )

  == Third and Fourth Features <third-fourth-features>

  Third and fourth features are sufficient to achieve a good accuracy with both the Multivariate and Tied Covariance Gaussian classifiers.

  We already expected this based again on @scatter, where we saw that the two features create two distinct clusters for each class.

  #figure(
    align(center)[
      #set par(justify: false)
      #table(
        columns: 3,
        align: center + horizon,
        table.hline(stroke: 0.5pt),
        inset: 1em,
        table.cell(fill: luma(250), []),
        table.cell(fill: luma(250), [*Accuracy*], inset: 1em),
        table.cell(fill: luma(250), [*Error Rate*], inset: 1em),
        table.hline(stroke: 0.5pt + gray),
        [*Multivariate*], [90.55%], [9.45%],
        [*Tied Covariance*], [90.60%], [9.40%],
        table.hline(stroke: 0.5pt),
      )
    ],
    caption: [Classification results for the third and fourth features],
  )

  == Reducing the Dimensionality with PCA

  We can try to reduce the dimensionality using PCA, we see that the `Multivariate Gaussian` model still performs the best from number of components $>= 4$, however the accuracy is still lower than the one obtained with the full dataset.

  #figure(
    image("imgs/pca_to_gaussians.svg"),
  )

  // #figure(
  //   align(center)[
  //     #set par(justify: false)
  //     #table(
  //       columns: 3,
  //       align: center + horizon,
  //       table.hline(stroke: 0.5pt),
  //       inset: 1em,
  //       table.cell(fill: luma(250), []),
  //       table.cell(fill: luma(250), [*Accuracy*], inset: 1em),
  //       table.cell(fill: luma(250), [*Error Rate*], inset: 1em),
  //       table.hline(stroke: 0.5pt + gray),
  //       [*Naïve Bayes*],
  //       [90.61%],
  //       [9.39%],
  //       table.hline(stroke: 0.5pt),
  //     )
  //   ],
  // )

  Applying PCA to reduce the dimensionality, doesn't improve the classification accuracy compared to the full dataset or the dataset without the last two features.

  == Conclusion

  Overall, the model that provided *the best* accuracy on the validation set is the `Multivariate Gaussian` model on the full dataset (@all-features). When reducing the dataset to 4 features the best strategy is to filter out the last two features and select the `Naïve Bayes` model (@filter-last-two). When reducing the dataset to 2 features the best strategy for now is using LDA (@lda-classifier[Lab 3 Section]).
]

#pagebreak(weak: true)

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
        table.cell(fill: luma(250), [#text(size: 1.2em, $bold(C_(f n))$)], inset: 1em),
        table.cell(fill: luma(250), [#text(size: 1.2em, $bold(C_(f p))$)], inset: 1em),
        table.cell(fill: luma(250), [#text(size: 1.2em, $bold(tilde(pi))$)], inset: 1em),
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

  == Best PCA Setup for $bold(tilde(pi)) = 0.1$

  We notice something interesting, if we isolate our analysis to the application with effective prior equal to `0.1` and only consider the `PCA` setups, we see that the optimal number of dimensions for the `Multivariate Gaussian` model is `6`, with a `minimum DCF` of `0.3015` and an `actual DCF` of `0.3182`, but the other variants prefer working on lower dimensions, even though we saw before that all three perform best without `PCA`. When looking at the `Tied Covariance Gaussian` and the `Naïve Bayes` models we see, respectively, that the optimal number of dimensions is `4`, with a `minimum DCF` of `0.3721` and an `actual DCF` of `0.3879` and the optimal number of dimensions is just `1`, with a `minimum DCF` of `0.3730` and an `actual DCF` of `0.4049`.

  Looking at the `Bayes Error Plots` for the three models in the prior log odds range $(-4, +4)$, we see that the models' rankings are moderately consistent across the applications and are overall well calibrated. We notice that the `Tied Covariance` one is ever so slightly better calibrated than the others but worse overall while the `Multivariate Gaussian` one is the best performing model overall.
]


#subpar.grid(
  figure(image("imgs/naive_prior_log_odds.svg"), caption: [Naïve Bayes]),
  figure(image("imgs/multivariate_prior_log_odds.svg"), caption: [Multivariate Gaussian]),
  figure(image("imgs/tied_prior_log_odds.svg"), caption: [Tied Covariance Gaussian]),
  columns: 3,
  caption: [Bayes Error Plots for the three best PCA models with $tilde(pi) = 0.1$],
)

= Logistic Regression

TODO

#pagebreak(weak: true)

= Support Vector Machines

#pagebreak(weak: true)

= Gaussian Mixture Models

#grid(
  columns: 2,
  figure(image("imgs/gmm/full.svg"), caption: [Full Covariance Matrix]),
  figure(image("imgs/gmm/diagonal.svg"), caption: [Diagonal Covariance Matrix]),
)
