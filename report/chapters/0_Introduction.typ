#import "@preview/oxifmt:0.2.0": strfmt
#import "../funcs.typ": eqcolumns

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
          image(strfmt("../imgs/scatter/overlay_{}_{}.svg", i, j)),
        ),
      )
    } else {
      cells.push(
        table.cell(
          inset: 0em,
          x: i + 1,
          y: j + 1,
          image(strfmt("../imgs/hist/histograms_{}.svg", i)),
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
