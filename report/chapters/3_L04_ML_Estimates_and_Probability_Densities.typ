#import "../funcs.typ": eqcolumns


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
    table.cell(inset: 0em, image("../imgs/densities/density_0_0.svg")),
    table.cell(inset: 0em, image("../imgs/densities/density_0_1.svg")),
    table.cell(inset: 0em, image("../imgs/densities/density_0_2.svg")),
    table.cell(inset: 0em, image("../imgs/densities/density_0_3.svg")),
    table.cell(inset: 0em, image("../imgs/densities/density_1_0.svg")),
    table.cell(inset: 0em, image("../imgs/densities/density_1_1.svg")),
    table.cell(inset: 0em, image("../imgs/densities/density_1_2.svg")),
    table.cell(inset: 0em, image("../imgs/densities/density_1_3.svg")),
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
      table.cell(inset: 0em, image("../imgs/densities/density_0_4.svg")),
      table.cell(inset: 0em, image("../imgs/densities/density_0_5.svg")),
      table.cell(inset: 0em, image("../imgs/densities/density_1_4.svg")),
      table.cell(inset: 0em, image("../imgs/densities/density_1_5.svg")),
    ),
  )

  When looking at features 5 and 6, on the other hand, we can see that the uni-variate Gaussian model does not fit the data well. The `Genuine` class in particular has a bimodal distribution for both features so it results in a particularly bad fit.

  == Maximum Likelihood Estimates

  $
    mu_("ML") = 1 / N sum_(i=1)^N x_i, space space sigma^2_("ML") = 1 / N sum_(i=1)^N (
      x_i - mu_("ML")
    )^2
  $

  The ML estimates for the parameters of a `Uni-variate Gaussian` model correspond to the dataset mean and variance for each feature.

  The following table summarizes the ML estimates for the dataset features.

  #table(
    columns: (auto, 1fr, 1fr),
    align: center + horizon,
    table.hline(stroke: 0.5pt),
    table.cell(
      fill: luma(250),
      colspan: 3,
      [#text(size: 1.2em, $bold(mu_("ML"))$)],
      inset: 1em,
    ),
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
    table.cell(
      fill: luma(250),
      colspan: 3,
      [#text(size: 1.2em, $bold(sigma^2_("ML"))$)],
      inset: 1em,
    ),
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
