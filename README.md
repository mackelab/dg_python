# dg_python
Dichotomized Gaussian model implemented using numpy

This repository mirrors code from the [CorBinian toolbox](https://github.com/mackelab/CorBinian) for fitting and sampling from dichotomized Gaussian models

*Note : CorBinian is written for MATLAB. It contains code for maximum entropy modeling and specific heat analysis, in addition to dichotomized Gaussian models for binary and integer count data, and accounting for spatio-temporal correlations. Here we **only** implement dichotomized Gaussian models binary count data.*


To install, clone the repository and run *pip install dg_python* from within the local directory containing the repository.

### Demo-scripts:
*dg_demo_fixed_frate.ipynb*: Sample spike trains with fixed firing rate per neurons from DG model; fit a DG model to this data.

*demo_dg_timevar_frate.ipynb* Sample spike trains with time-varying firing rate for every neuron from DG model; fit a DG model to this data.

### Package files
*dg_python/dichot_gauss*: python classes for sampling binary data from a DG model, and for the Higham algorithm (*NJ Higham, Computing the nearest correlation matrix - a problem from finance, IMA Journal of Numerical Analysis, 2002*)

*dg_python/optim_dichot_gauss*: python classes for fitting a DG model to binary data

### Publications
This repository contains code that implements methods outlined in

#####  [JH Macke*, P Berens*, AS Ecker, AS Tolias and M Bethge: Generating Spike Trains with Specified Correlation Coefficients. Neural Computation 21(2), 397-423, 02 2009](https://www.mitpressjournals.org/doi/10.1162/neco.2008.02-08-713)
Implemented here: functions for fitting and sampling from dichotomized Gaussian models with binary spike counts and fixed firing rate.

#####  [DR Lyamzin, JH Macke, NA Lesica: Modeling population spike trains with specified time-varying spike rates, trial-to-trial variability, and pairwise signal and noise correlations. Frontiers in Computational Neuroscience, 4 (144), pp. 1-11, 2010](https://www.mackelab.org/publications/#modeling-population-spike-trains-with-specified-time-varying-spike-rates-trial-to-trial-variability-and-pairwise-signal-and-noise-correlations)
Implemented here: functions for fitting and sampling from dichotomized Gaussian models with binary spike counts and time-varying firing rate.
