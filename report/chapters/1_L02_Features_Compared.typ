#import "../funcs.typ": eqcolumns

#set page(
  header: align(right, text(fill: gray)[`Report for lab 2`]),
)

= Features Compared
== Features 1 and 2 <features-1-2>
#grid(
  columns: 2,
  figure(image("../imgs/hist/histograms_0.svg"), caption: [Feature 1]),
  figure(image("../imgs/hist/histograms_1.svg"), caption: [Feature 2]),
)
#eqcolumns(2)[
  When looking at the first feature we can observe that the classes overlap almost completely. The `Genuine` label has a higher variance than the `Fake` class but the mean is similar. Both classes exhibit one mode in the histogram but the `Fake` class has a higher peak. Looking at the second feature we can notice the opposite behavior. The `Fake` class has a higher variance than the `Genuine` class but the mean is similar. Both classes exhibit one mode in the histogram but
  the `Genuine` class has a higher peak. Again, the classes overlap almost completely.
]

#line(length: 100%, stroke: 0.5pt + gray)

== Features 3 and 4
#grid(
  columns: 2,
  figure(image("../imgs/hist/histograms_2.svg"), caption: [Feature 3]),
  figure(image("../imgs/hist/histograms_3.svg"), caption: [Feature 4]),
)
#eqcolumns(2)[
  Looking at the plot for the third class we can notice that the two features are much more distinct, they overlap slightly in 0. The `Genuine` class has a peak in -1 while the `Fake` class has a peak in 1. They both have similar variance but the means differ. One mode for each class is evident from the histogram.
  The fourth feature shows similar characteristics to the third feature.
]

#pagebreak()

== Features 5 and 6
#grid(
  columns: 2,
  figure(image("../imgs/hist/histograms_4.svg"), caption: [Feature 5]),
  figure(image("../imgs/hist/histograms_5.svg"), caption: [Feature 6]),
)
#eqcolumns(2)[
  The fifth feature also shows a good distinction between the two classes with an overlap at the edges of the `Fake` class distribution. They exhibit similar variance but with a lower mean for the `Genuine` class. The `Fake` class peaks in 0 while the `Genuine` has two modes and peaks in -1 and 1.
  The last feature shows similar characteristics to the fifth feature.
]

#line(length: 100%, stroke: 0.5pt + gray)

#figure(
  image("../imgs/scatter/overlay_4_5.svg"),
  caption: [Features 5 and 6 Scatter Plot],
) <features-5-6-scatter>

Looking at the scatter plot we see that there are four distinct clusters for each of the labels, they overlap slightly at the edges of each cluster.
