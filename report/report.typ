#import "@preview/oxifmt:0.2.0": strfmt

#let title = [Report for the Machine Learning \ & Pattern Recognition Project]
#let author = (
  name: "Eduard Antonovic Occhipinti",
  id: 947847
)

#set document(
  title: title,
  author: author.name
)

#set text(
  font: "New Computer Modern",
  lang: "en",
  size: 10pt
)

#set page(
  paper: "a4",
  numbering: ("1"),
)

#show figure.caption: set text(size: 0.8em)

#set enum(
  indent: 10pt,
  body-indent: 9pt
)

#set list(
  indent: 10pt,
  body-indent: 9pt,
  marker: ([•], [--])
)

#show heading.where(level: 1): it => block(width: 100%, height: 20pt)[
  #set align(center)
  #set text(weight: "bold")
  #smallcaps(it)
]
#show heading.where(level: 2): it => block(width: 100%)[
  #set align(center)
  #set text(weight: "bold")
  #smallcaps(it)
]
#show heading.where(level: 3): it => block(width: 100%)[
  #set align(left)
  #set text(weight: "bold")
  #smallcaps(it)
]

#show raw.where(block: true): set text(size: 0.8em, font: "Fira Code")
#show raw.where(block: true): set par(justify: false)
#show raw.where(block: true): block.with(
  fill: gradient.linear(luma(240), luma(245), angle: 270deg),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)
#show raw.where(block: false): box.with(
  fill: gradient.linear(luma(240), luma(245), angle: 270deg),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 2pt,
)

#set par(justify: true)

#set align(center)

#line(length: 100%)

#block()[
  #smallcaps(
    text(
      size: 1.5em,
      weight: "bold",
      title
    )
  )
]

#line(length: 100%)

#block()[
  #author.name \
  #author.id
]

#v(40pt)

#set align(left)

#let graphs = ()
#for i in range(0, 6) {
  for j in range(0, 6) {
    if (i != j) { 
      graphs.push(
        grid.cell(
          x: i,
          y: j,
          image(strfmt("imgs/scatter/overlay_{}_{}.svg", i, j))
        )
      )
    } else {
      graphs.push(
        grid.cell(
          x: i,
          y: j,
          image(strfmt("imgs/hist/histograms_{}.svg", i)))
      )
    }
  }
}

#v(1fr)

#box(height: 90pt,
  columns(2)[
    = Introduction

    The task consists of a binary classification problem, the goal is to perform fingerprint spoofing detection (i.e. to distinguish between real and fake fingerprints). 
    
    #box(height: 37pt)
    The dataset consists of 6 features. In this first part, we will analyze some statistics of the dataset and the correlation between the features.
  ]
)

#grid(
  columns: 6,
  rows: 6,
 ..graphs
)

#v(1fr)

#pagebreak()

#columns(2)[
  == Features Compared

  === Features 1 and 2

  #image("imgs/hist/histograms_0.svg")

  When looking at the first feature we can observe that the classes overlap almost completely. The `Genuine` label has a higher variance than the `Fake` class but the mean is similar. Both classes exhibit one mode in the histogram but the `Fake` class has a higher peak.

  #image("imgs/hist/histograms_1.svg")

  Looking at the second feature we can notice the opposite behavior. The `Fake` class has a higher variance than the `Genuine` class but the mean is similar. Both classes exhibit one mode in the histogram but 
  #v(400pt)
  the `Genuine` class has a higher peak. Again, the classes overlap almost completely.

  // #box(height: 10pt)
  === Features 3 and 4

  #image("imgs/hist/histograms_2.svg")

  Looking at the plot for the third class we can notice that the two features are much more distinct, they overlap slightly in 0. The `Genuine` class has a peak in -1 while the `Fake` class has a peak in 1. They both have similar mean and variance. One mode for each class is evident from the histogram.

  #image("imgs/hist/histograms_3.svg")

  The fourth feature shows similar characteristics to the third feature. 
]

#pagebreak()

#columns(2)[
  === Features 5 and 6

  #image("imgs/hist/histograms_4.svg")

  The fifth feature also shows a good distinction between the two classes with an overlap at the edges of the `Fake` class distribution. They exhibit similar variance but with a lower mean for the `Genuine` class. The `Fake` class peaks in 0 while the `Genuine` has two modes and peaks in -1 and 1.

  #image("imgs/hist/histograms_5.svg")

  The last feature shows similar characteristics to the fifth feature.

  #image("imgs/scatter/overlay_4_5.svg")

  Looking at the scatter plot we see that there are four distinct clusters for each of the labels, they overlap slightly at the edges of each cluster.

  == Principal Component Analysis

  #grid(
    columns: 2,
    rows: 3,
    image("imgs/hist/pca/histograms_0.svg"),
    image("imgs/hist/pca/histograms_1.svg"),
    image("imgs/hist/pca/histograms_2.svg"),
    image("imgs/hist/pca/histograms_3.svg"),
    image("imgs/hist/pca/histograms_4.svg"),
    image("imgs/hist/pca/histograms_5.svg")
  )

  Looking at the principal components of the dataset we can see that only one results in a clear separation between the two classes and it seems to separate the two classes better than any other feature taken individually.

  == Linear Discriminant Analysis

  #image("imgs/hist/lda/histograms_0.svg")

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
      X_train_lda[y_train == 0].mean() + X_train_lda[y_train == 1].mean()
  ) / 2.0

  # Predict the validation data
  y_pred = [0 if x >= threshold else 1 for x in X_val_lda.T[0]]

  print(f"Threshold: {threshold:.2f}")
  print(f"Error rate: {np.sum(y_val != y_pred) / y_val.size * 100:.2f}%")
  ```

  ```
  Threshold: -0.02
  Error rate: 9.60%
  ```

  Empirically we can find that threshold `0.04` gives a slightly better error rate of `9.34%`.

  === Pre-processing the Data with PCA

  #image("imgs/error_rate_pca.svg")

  As we can see from the graph, pre-processing the data with PCA proves useful in reducing the error rate of the classifier slightly, in particular when choosing a number `N` of components equal to 2.
]
