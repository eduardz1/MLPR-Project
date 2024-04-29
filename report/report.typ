#import "@preview/oxifmt:0.2.0": strfmt
#import "template.typ": *
#import "funcs.typ": *

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
)

#v(1fr)

#set page(
  header: align(right, text(fill: gray)[`Report for lab 2`]),
)

= Features Compared
== Features 1 and 2
#grid(
  columns: 2,
  figure(image("imgs/hist/histograms_0.svg"), caption: [Feature 1]),
  figure(image("imgs/hist/histograms_1.svg"), caption: [Feature 2]),
)
#eqcolumns(2)[
  When looking at the first feature we can observe that the classes overlap almost completely. The `Genuine` label has a higher variance than the `Fake` class but the mean is similar. Both classes exhibit one mode in the histogram but the `Fake` class has a higher peak.

  Looking at the second feature we can notice the opposite behavior. The `Fake` class has a higher variance than the `Genuine` class but the mean is similar. Both classes exhibit one mode in the histogram but
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
  Looking at the plot for the third class we can notice that the two features are much more distinct, they overlap slightly in 0. The `Genuine` class has a peak in -1 while the `Fake` class has a peak in 1. They both have similar mean and variance. One mode for each class is evident from the histogram.

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

#eqcolumns(2)[
  == Principal Component Analysis

  #figure(
    caption: [Principal components computed by PCA, ordered from top to bottom, right to left],
    grid(
      columns: 2,
      rows: 3,
      image("imgs/hist/pca/histograms_0.svg"),
      image("imgs/hist/pca/histograms_1.svg"),
      image("imgs/hist/pca/histograms_2.svg"),
      image("imgs/hist/pca/histograms_3.svg"),
      image("imgs/hist/pca/histograms_4.svg"),
      image("imgs/hist/pca/histograms_5.svg"),
    ),
  )

  Looking at the principal components of the dataset we can see that only one results in a clear separation between the two classes and it seems to separate the two classes better than any other feature taken individually.

  == Linear Discriminant Analysis

  #figure(caption: [First LDA direction], image("imgs/hist/lda/histograms_0.svg"))

  We see that compared to the first principal the classes are mirrored but the separation is similar between the two methods.

  === Applying LDA as a classifier

  We now try to apply LDA as a classifier, we start by splitting the dataset in a training and validation set, then we fit the model on the training and evaluation set, then we calculate the optimal threshold for the classifier and finally, we evaluate the model on the validation set.

  ```python
  # Split the dataset into training and validation sets
  X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=0.33, random_state=0
  )

  # Fit the LDA model
  _, X_train_lda = lda(X_train, y_train, 1)
  _, X_val_lda = lda(X_val, y_val, 1)

  threshold = (
      X_train_lda[y_train == 0].mean() +
      X_train_lda[y_train == 1].mean()
  ) / 2.0

  # Predict the validation data
  y_pred = [
    0 if x >= threshold else 1 for x in X_val_lda.T[0]
  ]

  print(f"Threshold: {threshold:.2f}")
  print(f"Error rate: {
    np.sum(y_val != y_pred) / y_val.size * 100:.2f
  }%")
  ```

  ```
  Threshold: -0.02
  Error rate: 9.60%
  ```

  Empirically we can find that threshold `0.04` gives a slightly better error rate of `9.34%`.

  === Pre-processing the Data with PCA

  #figure(
    caption: [Error rates in percentage as a function of the number of LDA directions],
    image("imgs/error_rate_pca.svg"),
  )

  As we can see from the graph, pre-processing the data with PCA proves useful in reducing the error rate of the classifier slightly, in particular when choosing a number `N` of components equal to 2.
]

#pagebreak()

#set page(
  header: align(right, text(fill: gray)[`Report for lab 4`]),
)

= ML Estimates & Probability Densities

== Gaussian Models
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
    [#text(size: 0.8em, $2.87744301 * 10^(-3)$)],
    [#text(size: 0.8em, $5.44547838 * 10^(-4)$)],
    [2],
    [#text(size: 0.8em, $1.86931579 * 10^(-2)$)],
    [#text(size: 0.8em, $-8.52437392 * 10^(-3)$)],
    [3],
    [#text(size: 0.8em, $-6.80940159 * 10^(-1)$)],
    [#text(size: 0.8em, $6.65237846 * 10^(-1)$)],
    [4],
    [#text(size: 0.8em, $6.70836195 * 10^(-1)$)],
    [#text(size: 0.8em, $-6.64195349 * 10^(-1)$)],
    [5],
    [#text(size: 0.8em, $2.79569669 * 10^(-2)$)],
    [#text(size: 0.8em, $-4.17251858 * 10^(-2)$)],
    [6],
    [#text(size: 0.8em, $-5.82740035 * 10^(-3)$)],
    [#text(size: 0.8em, $2.39384879 * 10^(-2)$)],
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
  #table(
    columns: 4,
    align: center + horizon,
    table.hline(stroke: 0.5pt),
    inset: 1em,
    table.cell(fill: luma(250), []),
    table.cell(fill: luma(250), [*Multi Variate Gaussian*], inset: 1em),
    table.cell(fill: luma(250), [*Tied Gaussian*], inset: 1em),
    table.cell(fill: luma(250), [*Naive Bayes*], inset: 1em),
    table.hline(stroke: 0.5pt + gray),
    [*Accuracy*],
    [92.47%],
    [90.35%],
    [92.37%],
    [*Error Rate*],
    [7.53%],
    [9.65%],
    [7.63%],
    table.hline(stroke: 0.5pt),
  )
]