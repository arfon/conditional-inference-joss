Changelog
=========

1.0.0
-----

- Added nonparametric empirical Bayes
- Collapsed all normal prior Bayesian estimators (classic, maximum likelihood, and James-Stein) into a single ``Normal`` class
- Created a separate ``Improper`` class for Bayesian models with an improper prior
- Moved projection confidence intervals from ``RQU`` to a separate ``condfidence_set.ConfidenceSet`` class
- Created the ``confidence_set`` module
- Added non-parametric, mixture, and joint Distributions
- Added significance conditional analysis
- Renamed ``RQU`` to the more expressive ``RankCondition``

0.0.3
-----

- Tested on Python3.8
- Added Holm-Bonferroni correction
- Added ability to specify columns in ``ModeBase.from_csv``
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