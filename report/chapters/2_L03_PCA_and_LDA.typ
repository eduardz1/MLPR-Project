#import "../funcs.typ": eqcolumns

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
      image("../imgs/hist/pca/histograms_0.svg"),
      image("../imgs/hist/pca/histograms_1.svg"),

      image("../imgs/hist/pca/histograms_2.svg"),
      image("../imgs/hist/pca/histograms_3.svg"),

      image("../imgs/hist/pca/histograms_4.svg"),
      image("../imgs/hist/pca/histograms_5.svg"),
    ),
  )

  Looking at the principal components of the dataset we can see that only one results in a clear separation between the two classes and it seems to separate the two classes better than any other feature taken individually. The rotation makes it harder to see the clusters, if we were to plot the data in 6D we would see the same original clusters.

  == Linear Discriminant Analysis

  #figure(
    caption: [First LDA direction],
    image("../imgs/hist/lda/histograms_0.svg"),
  )

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
    image("../imgs/error_rate_pca.svg"),
  )

  As we can see from the graph, pre-processing the data with PCA does not improve the classification results. With the number of dimensions set to 2, the accuracy even decreases slightly.
]
