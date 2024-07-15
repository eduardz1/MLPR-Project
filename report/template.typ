#import "funcs.typ": clean_numbering

#let template(
  title: none,
  author: (),
  academic_year: "",
  body,
) = {
  set document(title: title, author: author.name)
  set text(font: "New Computer Modern", lang: "en", size: 10pt)
  set page(paper: "a4", numbering: ("1"))
  show figure.caption: set text(size: 0.8em, style: "italic")

  show link: underline

  set heading(numbering: clean_numbering("I -", "1.a."))
  show heading: it => if it.level != 1 {
    block(width: 100%, above: 2em, below: 1em, breakable: false)[
      #set par(justify: false)
      #set text(weight: "bold")
      #if it.numbering != none {
        grid(
          columns: 2,
          gutter: 0.5em,
          counter(heading).display(it.numbering), smallcaps(it.body),
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
          counter(heading).display(it.numbering), smallcaps(it.body),
        )
      } else {
        smallcaps(it)
      }
    ]
  }

  set table(stroke: none)

  show raw.where(block: true): set text(size: 0.8em)
  show raw.where(block: true): set par(justify: false)
  show raw.where(block: true): block.with(
    fill: luma(250),
    inset: 10pt,
    width: 100%,
    stroke: (top: 0.5pt, bottom: 0.5pt),
  )

  // Outline customization
  show outline.entry.where(level: 1): it => {
    if it.body != [References] {
      v(12pt, weak: true)
      link(
        it.element.location(),
        strong({
          it.body
          h(1fr)
          it.page
        }),
      )
    } else {
      text(size: 1em, it)
    }
  }

  // Title Page


  // Legend
  set align(center)
  set page(header: [
    #set align(right)
    #rect(stroke: 0.5pt + gray)[
      #grid(
        align: left,
        gutter: 0.5em,
        columns: 2,
        rect(fill: red.transparentize(30%), width: 30pt, height: 1em), [Fake],
        rect(fill: blue.transparentize(30%), width: 30pt, height: 1em),
        [Genuine],
      )
    ]
  ])

  line(length: 100%)

  // Title
  block()[
    #smallcaps(
      text(
        size: 1.6em,
        weight: "bold",
        title,
      ),
    )
  ]

  line(length: 100%)


  v(1fr)

  // Author and Academic Year
  block()[
    *#author.name* \
    #author.id \ \
    _Academic Year #academic_year _
  ]

  v(1fr)

  set align(left)

  set par(justify: true)

  body

  set page(header: none)

  bibliography("works.bib", title: "External References", full: true)
}
