.. Conditional Inference documentation master file, created by
   sphinx-quickstart on Mon Nov 12 14:17:27 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Multiple Inference documentation
================================

A statistics package for comparing multiple parameters (e.g., multiple treatments, policies, or subgroups).

.. image:: https://readthedocs.org/projects/dsbowen-conditional-inference/badge/?version=latest
   :target: https://dsbowen-conditional-inference.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://gitlab.com/dsbowen/conditional-inference/badges/master/pipeline.svg
   :target: https://gitlab.com/dsbowen/conditional-inference/-/commits/master
.. image:: https://gitlab.com/dsbowen/conditional-inference/badges/master/coverage.svg
   :target: https://gitlab.com/dsbowen/conditional-inference/-/commits/master
.. image:: https://badge.fury.io/py/conditional-inference.svg
   :target: https://badge.fury.io/py/conditional-inference
.. image:: https://img.shields.io/badge/License-MIT-brightgreen.svg
   :target: https://gitlab.com/dsbowen/conditional-inference/-/blob/master/LICENSE
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gl/dsbowen%2Fconditional-inference/HEAD?urlpath=lab
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

|
Motivation
==========

Multiple inference techniques outperform standard methods like OLS and IV estimation for comparing multiple parameters. For example, `this post <https://gitlab.com/dsbowen/conditional-inference/-/blob/master/examples/bayes_primer.ipynb>`_ shows how to apply Bayesian estimators to a randomized control trial testing many interventions to increase vaccination rates.

Start here
==========

Click the badges below to launch a Jupyter Binder with a ready-to-use virtual environment and template code.

This binder is an 80-20 solution for multiple inference.

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gl/dsbowen%2Fconditional-inference/HEAD?urlpath=lab/tree/examples/multiple_inference.ipynb

| This binder is for inference after ranking.

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gl/dsbowen%2Fconditional-inference/HEAD?urlpath=lab/tree/examples/rank_conditions.ipynb


|
Installation
============

.. code-block::

   $ pip install conditional-inference

Issues
======

Please submit issues `here <https://gitlab.com/dsbowen/conditional-inference/-/issues>`_.

Contents
========

.. toctree::
   :maxdepth: 2

   conditional_inference/index
   Changelog <changelog>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citations
=========

.. code-block::

   @software(multiple-inference,
      title={ Multiple Inference },
      author={ Bowen, Dillon },
      year={ 2022 },
      url={ https://dsbowen-conditional-inference.readthedocs.io/en/latest/?badge=latest }
   )

Acknowledgements
================

I would like to thank Isaiah Andrews, Toru Kitagawa, Adam McCloskey, and Jeff Rowley for invaluable feedback on my early drafts.

My issue templates are based on the `statsmodels <https://github.com/statsmodels/statsmodels/issues/new/choose>`_ issue templates.