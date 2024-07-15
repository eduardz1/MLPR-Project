#import "template.typ": *

#show: doc => template(
  title: [Report for the Machine Learning \ & Pattern Recognition Project],
  author: (
    name: "Eduard Antonovic Occhipinti",
    id: 947847,
  ),
  academic_year: "2023/2024",
  doc,
)

#include "chapters/0_Introduction.typ"

#include "chapters/1_L02_Features_Compared.typ"

#include "chapters/2_L03_PCA_and_LDA.typ"

#include "chapters/3_L04_ML_Estimates_and_Probability_Densities.typ"

#include "chapters/4_L05_Generative_Models_for_Classification.typ"

#include "chapters/5_L07_Performance_Analysis_of_the_MVG_Classifier.typ"

#include "chapters/6_L08_Logistic_Regression.typ"

#include "chapters/7_L09_Support_Vector_Machines.typ"

#include "chapters/8_L10_Gaussian_Mixture_Models.typ"

#include "chapters/9_L11_Calibration_and_Evaluation.typ"
