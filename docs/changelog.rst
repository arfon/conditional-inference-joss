Changelog
=========

0.0.3
-----

- Fixed a column selection bug in ``ProjectionResults.conf_int``
- Fixed a bug in Bayes results base: allows for rank matrix when estimating one parameter
- Fixed a bug in ``RQU.get_distributions``; the projection CIs didn't line up with the column indices in the previous version

0.0.2
-----

- Added a ``truncnorm`` distribution with exponential tilting
- Added reconstruction plot methods for Bayesian models
- Added utilities for Wasserstein distances as a measure of reconstruction error
- Added robustness to linear empirical Bayes likelihood optimization
- Added empirical Bayes fitting to minimize Wasserstein reconstruction error
- Changed the prior covariance parameter for empirical and hierarchical Bayes to represent the prior standard deviation rather than the prior variance
- Fixed a bug in rank matrix calculation

0.0.1
-----

- First release on PyPI