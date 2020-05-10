# Datasets for causal inference (ITE) and feature selection

## IHDP
The Infant Health and Development Program [[1]](#1) was an RCT (n = 985, control = 608, treated = 377) with binary treatment assignment and 28 recorded covariates (binary = 22, continuous = 6). Following Hill [[2]](#2), we induce selection bias by removing non-whites from the treated population (such that treated = 139) and discard the 3 binary covariates describing race. We then generate synthetic outcomes according to one of two settings:

![Y \sim N\left(\begin{bmatrix}X\beta_A\\X\beta_A + 4\end{bmatrix},\,I\right)](https://render.githubusercontent.com/render/math?math=Y%20%5Csim%20N%5Cleft(%5Cbegin%7Bbmatrix%7DX%5Cbeta_A%5C%5CX%5Cbeta_A%20%2B%204%5Cend%7Bbmatrix%7D%2C%5C%2CI%5Cright))

![Y \sim N\left(\begin{bmatrix}\exp((X+W)\beta_B)\\X\beta_B\end{bmatrix},\,I\right)](https://render.githubusercontent.com/render/math?math=Y%20%5Csim%20N%5Cleft(%5Cbegin%7Bbmatrix%7D%5Cexp((X%2BW)%5Cbeta_B)%5C%5CX%5Cbeta_B%5Cend%7Bbmatrix%7D%2C%5C%2CI%5Cright))

where the &beta; are randomly generated, taking values in {0, 1, 2, 3, 4} and {0, 0.1, 0.2, 0.3, 0.4}, respectively. This means it is likely that some covariates are not used in the generative model.

## LBIDD
The Linked Births and Infant Deaths Database (LBIDD) [[3]](#3) was used in the IBM Causal Inference Benchmarking Framework [[4]](#4) to provide an extensive suite of experimental settings, with sample sizes ranging from 1000 to 50000, and datasets with various levels of censoring, imbalance, and numerous generative models. These all stem from a shared covariates file consisting of 177 features (73 binary, 58 ternary, 46 continuous).

## Jobs
3212 = 2915 control + 297 treated,
7 covariates (binary = 4, continuous = 3).

## Twins

## L2X
**Syn1-6** inspired by L2X [[5]](#5) and INVASE [[6]](#6) papers, implemented in *synthetic_data.py*.

## References
<a id="1">[1]</a> 
C. T. Ramey, D. M. Bryant, B. H. Wasik, J. J. Sparling, K. H. Fendt, and L. M. LaVange.
Infant Health and Development Program for low birth weight, premature infants: program elements, family participation, and child intelligence.
Pediatrics, 89(3):454–465, Mar 1992.

<a id="2">[2]</a>
Jennifer L. Hill.
Bayesian nonparametric modeling for causal inference.
Journal of Computational and Graphical Statistics, 20(1):217–240, 2011.

<a id="3">[3]</a>
Marian F MacDorman and Jonnae O Atkinson.
Infant mortality statistics from the linked birth/infant death data set - 1995 period data.
Mon Vital Stat Rep, 46(suppl 2):1-22, 1998.

<a id="4">[4]</a>
Ehud Karavani, Yishai Shimoni, & Chen Yanover. (2018, January 31). 
IBM Causal Inference Benchmarking Framework (Version v1.0.0). 
Zenodo. http://doi.org/10.5281/zenodo.1163587

<a id="5">[5]</a>
Jianbo Chen, Le Song, Martin J Wainwright, and Michael I Jordan.
Learning to explain: An information-theoretic perspective on model interpretation.
arXiv preprint arXiv:1802.07814, 2018.

<a id="6">[6]</a>
Jinsung Yoon, James Jordon, and Mihaela van der Schaar.
INVASE: Instance-wise variable selection using neural networks.
In International Conference on Learning Representations, 2019.
