#import "funcs.typ": *

#let template(
  title: [Report for the Machine Learning \ & Pattern Recognition Project],
  author: (
    name: "Eduard Antonovic Occhipinti",
    id: 947847,
  ),
  body,
) = {
  set document(title: title, author: author.name)
  set text(font: "New Computer Modern", lang: "en", size: 10pt)
  set page(paper: "a4", numbering: ("1"))
  show figure.caption: set text(size: 0.8em)
  set enum(indent: 10pt, body-indent: 9pt)
  set list(indent: 10pt, body-indent: 9pt, marker: ([•], [--]))
 show figure.caption: emph

  set heading(numbering: clean_numbering("I -", "1.a."))
  show heading: it => if it.level != 1 {
    block(width: 100%, above: 2em, below: 1em, breakable: false)[
      #set par(justify: false)
      #set text(weight: "bold")
      #if it.numbering != none {
        grid(
          columns: 2,
          gutter: 0.5em,
          counter(heading).display(it.numbering),
          smallcaps(it.body),
        )
      } else {
        smallcaps(it)
      }
    ]
  } else {
    pagebreak(weak: true)
    block(width: 100%, above: 3em, below: 3em, breakable: false)[
      #set par(justify: false)
      #set align(center)
      #set text(weight: "bold")
      #if it.numbering != none {
        grid(
          columns: 2,
          gutter: 0.5em,
          counter(heading).display(it.numbering),
          smallcaps(it.body),
        )
      } else {
        smallcaps(it)
      }
    ]
  }


  show raw.where(block: true): set text(size: 0.7em, font: "Fira Code")
  show raw.where(block: true): set par(justify: false)
  show raw.where(block: true): block.with(
    fill: luma(250),
    inset: 10pt,
    width: 100%,
    stroke: (top: 0.5pt, bottom: 0.5pt),
  )

  set align(center)

  line(length: 100%)

  block()[
    #smallcaps(
      text(
        size: 1.5em,
        weight: "bold",
        title,
      ),
    )
  ]

  line(length: 100%)

  [
    #set align(right)
    #grid(
      align: left,
      gutter: 0.5em,
      columns: 2,
      rect(fill: red.transparentize(30%), width: 30pt, height: 1em),
      [Fake],
      rect(fill: blue.transparentize(30%), width: 30pt, height: 1em),
      [Genuine],
    )
  ]

  block()[
    *#author.name* \
    #author.id
  ]

  v(1fr)

  set align(left)

  set par(justify: true)

  body
}