#import "../funcs.typ": eqcolumns


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
      image("../imgs/heatmaps/covariance_fake.svg"),
      image("../imgs/heatmaps/covariance_genuine.svg"),
    ),
  )

  We notice the covariance values are very small compared to the variances. To better visualize the strength of the variances with respect to the covariances we can compute the correlation matrices for the two classes.

  #figure(
    caption: [Correlation matrices for the `Fake` and `Genuine` classes],
    grid(
      columns: 2,
      inset: (right: -5pt, left: -5pt),
      image("../imgs/heatmaps/correlation_fake.svg"),
      image("../imgs/heatmaps/correlation_genuine.svg"),
    ),
  ) <correlation-matrices>

  From the correlation matrices, we can see that the features are weakly correlated with each other. Given the Naïve Bayes assumption of independence, $p(E | c) = product_(i = 1)^n p (x_i | c)$ where $E = (x_1, x_2, dots, x_n)$, it's not surprising that the model performs well. (Note that independence is not needed for the model to perform well, #cite(<Zhang2004TheOO>, form: "prose") explores the optimality of the Naïve Bayes classifier even in the presence of correlation between the features).

  == Filtering out the Last Two Features

  We noticed in @gaussian-models[Lab 4 Section] that the last two features do not fit well with the Gaussian assumption. When repeating the classification tasks without the last two features, we notice that the accuracy decreases slightly across the board. This result, perhaps unexpected, implies that there is still valuable information to be extracted from these features.

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

  We can try to reduce the dimensionality using PCA, we see that the `Multivariate Gaussian` model still performs the best from a number of components $>= 4$, however, the accuracy is still lower than the one obtained with the full dataset.

  #figure(
    image("../imgs/pca_to_gaussians.svg"),
  )

  Applying PCA to reduce the dimensionality, doesn't improve the classification accuracy compared to the full dataset or the dataset without the last two features.

  == Conclusion

  Overall, the model that provided *the best* accuracy on the validation set is the `Multivariate Gaussian` model on the full dataset (@all-features). When reducing the dataset to 4 features the best strategy is to filter out the last two features and select the `Naïve Bayes` model (@filter-last-two). When reducing the dataset to 2 features the best strategy for now is using LDA (@lda-classifier[Lab 3 Section]).
]
