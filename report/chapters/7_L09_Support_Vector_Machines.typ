#import "../funcs.typ": eqcolumns

#set page(
  header: align(right, text(fill: gray)[`Report for lab 9`]),
)

= Support Vector Machines

#eqcolumns(2)[
  == Linear SVM

  We start by analyzing the performance of the `Linear SVM` model. we notice that the `minimum DCF` if we exclude the case with the regularization term $C = 10^(-5)$, #footnote[No support vector has been found, i.e. $bold(alpha)^*$ = [0, ..., 0]] is consistent across our range of $C$ values. The `actual DCF`, on the other hand, is more affected by the regularization term $C$ and is inversely proportional to it.

  The model performs poorly compared to the ones we have analyzed before and remains poorly calibrated even with high values of $C$, the `actual DCF` seems to tend to a value of `0.5` for $C -> infinity$

  #figure(
    caption: [Effects of the regularization coefficient $C$ on the `DCF` for the `Linear SVM` model],
    image("../imgs/svm/linear.svg"),
  )

  === Centering the Dataset

  Again we notice very little difference by centering the dataset before training the model due to the dataset being already centered.


  == Polynomial Kernel SVM

  Repeating the analysis with the `Polynomial Kernel SVM` we notice a similar graph but with lower overall values of `DCF`. The degree 4 Polynomial Kernel performs much better out of the gate even though the `minimum DCF` goes over the one for the degree 2 kernel towards the end of our range of $C$ values. The degree 4 kernel is much better calibrated for our application prior.

  #figure(
    caption: [Effects of the regularization coefficient $C$ on the `DCF` for the `Polynomial Kernel SVM`],
    image("../imgs/svm/poly_kernel.svg"),
  )

  === Considerations on the degree

  Looking at the way features 5 and 6 are distributed (@features-5-6-scatter) we see that a second-degree polynomial transformation would only push the features higher in a third dimension but that would not enable us to find any new meaningful separation hyperplane between the two, thus a higher degree is needed.

  == RBF Kernel SVM

  The RBF Kernel proves to be the best out of the three variants, the best $gamma$ value is $e^(-2)$ with a $C$ of $approx 32$ for the best `minimum DCF` for $tilde(pi) = 0.1$, it's interesting seeing how different values of the hyperparameters affect the decision boundary, making it more or less smooth (in regards to $gamma$) and more or less complex (in regards to $C$).

  #figure(
    caption: [Effects of the regularization coefficient $C$ on the `DCF` for the `RBF Kernel SVM` model],
    image("../imgs/svm/rbf_kernel.svg"),
  )

]
