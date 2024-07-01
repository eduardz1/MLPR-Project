"""
# Analyzing the features

The project task consists of a binary classification problem. The goal is to
perform fingerprint spoofing detection, i.e. to identify genuine vs counterfeit
fingerprint images. The dataset consists of labeled samples corresponding to the
genuine (True, label 1) class and the fake (False, label 0) class. The samples
are computed by a feature extractor that summarizes high-level characteristics
of a fingerprint image. The data is 6-dimensional. The training files for the
project are stored in file `Project/trainData.txt`. The format of the file is
the same as for the Iris dataset, i.e. a csv file where each row represents a
sample. The first 6 values of each row are the features, whereas the last value
of each row represents the class (1 or 0). The samples are not ordered.
Load the dataset and plot the histogram and pair-wise scatter plots of the
different features. Analyze the plots.

-   Analyze the first two features. What do you observe? Do the classes overlap?
    If so, where? Do the classes show similar mean for the first two features?
    Are the variances similar for the two classes? How many modes are evident
    from the histograms (i.e., how many “peaks” can be observed)?
-   Analyze the third and fourth features. What do you observe? Do the classes
    overlap? If so, where? Do the classes show similar mean for these two
    features? Are the variances similar for the two classes? How many modes are
    evident from the histograms?
-   Analyze the last two features. What do you observe? Do the classes overlap?
    If so, where? How many modes are evident from the histograms? How many
    clusters can you notice from the scatter plots for each class?
"""

from project.figures.plots import hist, scatter
from project.funcs.base import load_data


def lab02(DATA: str):
    X, y = load_data(DATA)

    hist(X, y, file_name="histograms")

    scatter(X, y, file_name="overlay")
