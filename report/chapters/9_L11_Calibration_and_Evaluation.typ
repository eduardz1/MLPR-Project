#import "../funcs.typ": eqcolumns
#import "@preview/subpar:0.1.1"


#set page(
  header: align(right, text(fill: gray)[`Report for lab 11`]),
)

= Calibration and Evaluation

#grid(
  columns: 2,
  gutter: 1.9em,
  align: bottom,
  [
    == Calibration

    === Gaussian Mixture Model

    The best prior $pi$ for the calibration transformation was $0.8$ but given that the model was already very well calibrated, the improvement is negligible (`actual DCF` decreased only by $0.0009$).

    #figure(
      image("../imgs/calibration/gmm.svg"),
      caption: [Calibration transformation for the Gaussian Mixture Model],
    )],
  [
    === Logistic Regression

    For the `Logistic Regression` model the calibration, now with a prior $pi = 0.1$, was more beneficial.

    #v(26pt)

    #figure(
      image("../imgs/calibration/log_reg.svg"),
      caption: [Calibration transformation for the Logistic Regression Model],
    )

  ],

  [

    === Support Vector Machine

    The `Support Vector Machine` model, calibrated with $pi = 0.5$, is the one that has seen the biggest improvement compared to its previous iteration.

    #figure(
      image("../imgs/calibration/svm.svg"),
      caption: [Calibration transformation for the Support Vector Machine Model],
    )],
  [

    === Score Level Fusion

    The fusion of the model performs pretty well overall, with a performance that is very similar to the `Gaussian Mixture Model`. It's very well calibrated.

    #figure(
      image("../imgs/calibration/fusion.svg"),
      caption: [Calibration transformation for the Score level Fusion Model],
    )],
)

#pagebreak(weak: true)

=== Delivery Model

I chose the `Gaussian Mixture Model` as the delivery model given the shape of the dataset and the performance of the model. I suspect that the high number of components for the `True` class might lead to a slight overfitting of the training set but I will leave it as such because it's difficult to judge the real shape of the clusters in 6 dimensions.

#figure(
  image("../imgs/calibration/complete.svg"),
  caption: [Bayes error plot for all of the models and their fusion, after calibration],
)

#columns(2)[
  == Evaluation

  === Evaluation of the Delivered System

  Overall the delivered system performs well on the evaluation set, the `actual DCF` is `0.2073` while the `minimum DCF` is `0.1838` for the target prior $pi = 0.1$ so it's well calibrated.

  We notice from the `Bayes Error Plot` that the model is also well calibrated across a wide range of applications with just a small range where it could've been better calibrated.

  #figure(
    image("../imgs/evaluation/delivery.svg"),
    caption: [Bayes error plot for the delivered system],
  )
]

#pagebreak(weak: true)

#columns(2)[
  === `actual DCF` of the Four Calibrated Models

  Looking at the `Bayes Error Plot` of the actual DCF for the four models, we see that our choice was indeed the correct one.

  #figure(
    image("../imgs/evaluation/actDCF.svg"),
    caption: [Bayes error plot for the `actual DCf` for the four models, after calibration, evaluated on the evaluation set],
  )

  === `minimum DCF` and `actual DCF` on our Application $bold(tilde(pi) = 0.1)$ for the three Models

  As we can see, for our application prior the `Gaussian Mixture Model` is confirmed to be the best performing one.

  #[
    #set par(justify: false)
    #table(
      columns: 4,
      align: center + horizon,
      table.hline(stroke: 0.5pt),
      inset: 1em,
      table.cell(fill: luma(250), [*DCF*]),
      table.cell(fill: luma(250), [*LogReg*], inset: 1em),
      table.cell(fill: luma(250), [*SVM*], inset: 1em),
      table.cell(fill: luma(250), [*GMM*], inset: 1em),
      table.hline(stroke: 0.5pt + gray),
      [minimum], [0.3486], [0.2622], [0.1838],
      [actual], [0.4046], [0.2863], [0.2073],
      table.hline(stroke: 0.5pt),
    )]

  We can further validate our choice by looking at the `Bayes Error Plot` for the three.

  #figure(
    image("../imgs/evaluation/act_min_DCF.svg"),
    caption: [Comparison of the Bayes error plots for the best models for the application with effective prior $tilde(pi) = 0.1$ on the evaluation set],
  )

  We notice that for all three of the models, the calibration strategy was quite effective even though all three present a small mis-calibration on the same `Log Odds` range.

  #v(1fr)

  === Validation for Other GMMs

  Finally, we can perform validation on all the other combinations of `GMM` that we discarded.

  #subpar.grid(
    gutter: 1em,
    figure(
      table(
        columns: 8,
        align: center + horizon,
        inset: 0.6em,
        table.hline(stroke: 0.5pt),
        table.vline(stroke: 0.5pt + gray, x: 2, start: 2),
        table.cell(fill: luma(250), []),
        table.cell(fill: luma(250), []),
        table.cell(fill: luma(250), [$N^o_"negative"$], colspan: 6),
        table.cell(fill: luma(250), [], x: 0, y: 1),
        table.cell(
          fill: luma(250),
          rowspan: 6,
          rotate(-90deg, reflow: true)[$N^o_"positive"$],
          x: 0,
          y: 2,
        ),

        table.cell(fill: luma(250), []),
        table.cell(fill: luma(250), [*1*]),
        table.cell(fill: luma(250), [*2*]),
        table.cell(fill: luma(250), [*4*]),
        table.cell(fill: luma(250), [*8*]),
        table.cell(fill: luma(250), [*16*]),
        table.cell(fill: luma(250), [*32*]),

        table.cell(fill: luma(250), [*1*], x: 1, y: 2),
        table.cell(fill: luma(250), [*2*], x: 1, y: 3),
        table.cell(fill: luma(250), [*4*], x: 1, y: 4),
        table.cell(fill: luma(250), [*8*], x: 1, y: 5),
        table.cell(fill: luma(250), [*16*], x: 1, y: 6),
        table.cell(fill: luma(250), [*32*], x: 1, y: 7),

        table.hline(stroke: 0.5pt + gray, start: 2),

        [0.407], [0.200], [0.325], [0.257], [0.254], [0.316],
        [0.410], [0.297], [0.330], [0.253], [0.252], [0.315],
        [0.272], [0.236], [0.246], [0.211], [0.201], [0.254],
        [0.218], [0.212], [0.212], table.cell(
          fill: green.lighten(80%),
          [0.180],
        ), [0.188], [0.227],
        [0.233], [0.204], [0.215], [0.185], [0.202], [0.229],
        [0.235], [0.226], [0.231], [0.195], [0.196], [0.229],
        table.hline(stroke: 0.5pt),
      ),
      caption: [`Full Covariance`],
    ),

    figure(
      table(
        columns: 8,
        align: center + horizon,
        inset: 0.6em,
        table.hline(stroke: 0.5pt),
        table.vline(stroke: 0.5pt + gray, x: 2, start: 2),
        table.cell(fill: luma(250), []),
        table.cell(fill: luma(250), []),
        table.cell(fill: luma(250), [$N^o_"negative"$], colspan: 6),
        table.cell(fill: luma(250), [], x: 0, y: 1),
        table.cell(
          fill: luma(250),
          rowspan: 6,
          rotate(-90deg, reflow: true)[$N^o_"positive"$],
          x: 0,
          y: 2,
        ),

        table.cell(fill: luma(250), []),
        table.cell(fill: luma(250), [*1*]),
        table.cell(fill: luma(250), [*2*]),
        table.cell(fill: luma(250), [*4*]),
        table.cell(fill: luma(250), [*8*]),
        table.cell(fill: luma(250), [*16*]),
        table.cell(fill: luma(250), [*32*]),

        table.cell(fill: luma(250), [*1*], x: 1, y: 2),
        table.cell(fill: luma(250), [*2*], x: 1, y: 3),
        table.cell(fill: luma(250), [*4*], x: 1, y: 4),
        table.cell(fill: luma(250), [*8*], x: 1, y: 5),
        table.cell(fill: luma(250), [*16*], x: 1, y: 6),
        table.cell(fill: luma(250), [*32*], x: 1, y: 7),

        table.hline(stroke: 0.5pt + gray, start: 2),

        [0.410], [0.366], [0.250], [0.254], [0.277], [0.256],
        [0.412], [0.366], [0.251], [0.255], [0.279], [0.262],
        [0.277], [0.269], [0.195], [0.192], [0.208], [0.219],
        [0.277], [0.272], [0.194], [0.203], [0.211], [0.223],
        [0.214], [0.215], table.cell(
          fill: green.lighten(80%),
          stroke: 1pt + green,
          [0.178],
        ), table.cell(
          fill: green.lighten(80%),
          [0.181],
        ), [0.189], [0.192],
        [0.218], [0.217], table.cell(
          fill: green.lighten(80%),
          [0.183],
        ), table.cell(
          fill: red.lighten(80%),
          [0.184],
        ), [0.188], [0.196],
        table.hline(stroke: 0.5pt),
      ),
      caption: [`Diagonal Covariance`],
    ),
    caption: [`minimum DCF` on evaluation for all combinations of `GMM` models, in *RED* is our choice, in *GREEN* the models that score better than our delivery (highlighted one being the best)],
  )

  We notice that our choice turned out to be close to optimal, as I guessed, the combination of 8 negative and 32 positive components was a bit too much and it led to a slight overfitting of the training set, the difference from the optimal however is quite small.

  It's also interesting seeing that there is a combination of components so that the full covariance matrix variant performs better than our diagonal.

  #v(1fr)

]