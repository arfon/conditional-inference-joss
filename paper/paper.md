---
title: 'Multiple Inference: A Python package for comparing multiple parameters'
tags:
  - Python
  - multiple inference
  - conditional inference
  - post-selection inference
authors:
  - name: Dillon Bowen
    orcid: 0000-0002-3033-1332
    affiliation: 1
affiliations:
 - name: Wharton School of Business, University of Pennsylvania
   index: 1
date: 14 May 2022
bibliography: paper.bib
---

# Summary

Scientists often want to compare many parameters. For example, scientists often run randomized control trials to study the effects of many treatments, use observational data to compare many geographic regions, and study how public policy will impact many subgroups of people. Multiple Inference implements many of the latest econometric and statistical tools for making such comparisons, including inference after ranking, simultaneous confidence sets, and Bayesian estimators. It uses a `statsmodels`-like API and provides template notebooks for ease of use. In just a few clicks, researchers can upload a `.csv` file of conventional estimates (e.g., OLS or IV estimates) to a Jupyter binder and click "run" to apply a suite of multiple inference tools to their data.

# Statement of need

Researchers often want to compare multiple parameters. For example, there is a recent trend in social science to run large-scale studies and randomized control trials designed to test the effectiveness of many behavioral interventions [@cortese2019megastudy]. Researchers have used large-scale field studies to test the effectiveness of many text messages reminding patients to get vaccinated [@milkman2021megastudy; @milkman2022680; @banerjee2021selecting], behavioral nudges encouraging 24 Hour Fitness customers to exercise more often [@milkman2021megastudies], monetary and social incentives to exert effort [@dellavigna2018motivates], behavioral interventions to decrease implicit racial bias [@lai2014reducing], donation matching schemes to increase charitable giving [@karlan2007does], and job training programs to increase employment among refugees in Jordan [@caria2020adaptive]. Researchers also perform multiple comparisons using observational data. For example, economists often use observation data to compare many neighborhoods in terms of intergenerational mobility [@chetty2018opportunity; @chetty2018impacts; @chetty2014land].

Researchers tend to ask the same set of questions when comparing many parameters.

1. Which parameters are significantly different from zero?
2. Which parameters are significantly better than the average (i.e., the average value across all parameters)?
3. Which parameters are significantly different from which other parameters?
4. What is the ranking of each parameter?
5. Which parameters might be the largest (i.e., the highest-ranked)?
6. What are the values of the parameters given their rank? e.g., What is the value of the parameter with the largest estimated value?
7. How are the parameters distributed?

Researchers often use conventional estimators like ordinary least squares (OLS) and instrumental variables (IV) to answer such questions [@milkman2021megastudy; @milkman2022680; @milkman2021megastudies; @lai2014reducing]. Unfortunately, conventional estimators overestimate the value of the top-performing parameter (i.e., the parameter with the largest estimated value) and exaggerate the variability of the parameters [@andrews2019inference; @andrewsinference]. These problems lead researchers to overstate the effectiveness of the top-performing treatments and the differences between treatment effects.

Statisticians and econometricians have advanced multiple inference tools in recent years. Recent publications describe new statistical techniques for inference after ranking [@andrews2019inference; @andrewsinference], multiple hypothesis testing [@romano2005stepwise], rank estimation [@mogstad2020inference], and Bayesian estimation [@stein1956inadmissibility; @james1992estimation; @dimmery2019shrinkage; @cai2021nonparametric; @brown2009nonparametric]. However, these techniques are mathematically complex and often inaccessible to all but professional statisticians.

Multiple Inference solves this problem by implementing many of the latest multiple inference tools in an easy-to-use `statsmodels`-like API. Additionally, Multiple Inference provides Jupyter binders with boilerplate code and narrative explanations to help researchers interpret the output. These binders allow researchers to upload a `.csv` file of their conventional estimates and click "run" to apply multiple inference tools to their data without downloading any software or writing a single line of code.

Multiple Inference initially implemented the inference after ranking techniques in @andrews2019inference and extended in @andrewsinference. The latter paper uses Multiple Inference to compare many United States commuting zones regarding intergenerational mobility. The World Bank Group is currently using Multiple Inference to reanalyze the results of a multi-treatment study designed to improve tax collection in Poland (see @hernandez2017applying for an earlier version of the paper).

# State of the field

Multiple Inference's defining features are inference after ranking, rank estimation, and hypothesis testing tools. Most importantly, Multiple Inference contains the only implementation of the inference after ranking techniques developed in @andrews2019inference and @andrewsinference in any language. These techniques correct for the winner's curse when performing inference on top-performing parameters (e.g., the parameters that rank in the top five according to conventional estimates). Specifically, Multiple Inference implements computationally efficient algorithms for computing quantile-unbiased point estimates and confidence intervals with correct coverage for parameters of specific ranks.

@mogstad2020inference has an associated R package for estimating rankings. For example, it may estimate that a particular parameter has a 95% chance of being one of the three largest parameters. It also computes sets of parameters that contain all of the truly largest $K$ parameters with 95% confidence. Multiple Inference is the only Python implementation of these techniques.

`statsmodels` implements multiple testing corrections based on p-values, such as the Holm-Bonferroni correction. Multiple Inference implements multiple hypothesis tests using a more powerful stepdown method based on simultaneous confidence sets for jointly Gaussian distributed estimates [@romano2005stepwise].

Bayesian estimators are essential tools for multiple inference, and robust packages for Bayesian analysis already exist in Python. For example, `statsmodels` implements two Bayesian models (binomial and Poisson) with independent Gaussian priors [@seabold2010statsmodels]. `pymc3` is a comprehensive package for Bayesian inference [@salvatier2016probabilistic]. Additionally, @dimmery2019shrinkage implements a Gaussian prior Bayesian model fit using Stein-type estimation. It distinguishes itself by incorporating uncertainty about the estimates of the prior parameters into the posterior distribution.

Multiple Inference aims to be a one-stop-shop for multiple inference and therefore includes parametric and nonparametric Bayesian estimators. Its Gaussian prior Bayesian estimator is most similar to the Stein-type estimator from @dimmery2019shrinkage. However, @dimmery2019shrinkage does not account for correlated errors. For example, if we underestimate the prior mean and shrink all posterior estimates towards the estimated prior mean, we will underestimate many parameters. Multiple Inference accounts for this correlated uncertainty in its James-Stein fit method of the Gaussian prior Bayesian model. Additionally, Multiple Inference provides a maximum likelihood fit method for the Gaussian prior model, also accounting for correlated uncertainty about the prior parameters.

Multiple Inference also implements several "intermediate products" that researchers can use in other applications. The most notable is a truncated normal distribution with two advantages over `scipy`'s truncated normal distribution [@2020SciPy-NMeth]. First, `scipy`'s truncated normal required a convex truncation set, whereas Multiple Inference's truncated normal accepts both convex and concave truncation sets. Second, Multiple Inference uses an exponential tilting method to improve accuracy when the truncation set is far from the mean of the underlying normal [@botev2017normal; @botev2015efficient]. Multiple Inference uses the same exponential tilting method as the R package `TruncatedNormal` [@botev2021truncatednormal]. We can see the advantage of Multiple Inference's implementation for the cumulative distribution function of a standard normal truncated to the interval $[8, 9]$ evaluated at 8.7.

```python
>>> from scipy.stats import truncnorm
>>> truncnorm(8, 9).cdf(8.7)
1.0709836154559238
>>> from conditional_inference.stats import truncnorm
>>> truncnorm([(8, 9)]).cdf(8.7)
0.9978948153314305
```

# Acknowledgements

I would like to thank Sarah Reed and Christian Kaps for feedback on this paper. I would also like to thank Isaiah Andrews, Toru Kitagawa, Adam McCloskey, and Jeff Rowley for feedback on my early drafts of the software.

# References