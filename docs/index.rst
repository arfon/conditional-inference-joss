.. Conditional Inference documentation master file, created by
   sphinx-quickstart on Mon Nov 12 14:17:27 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Conditional Inference documentation
===========================================

A statistics package for comparing multiple policies or treatments.

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
Quickstart
==========

Click the badges below to launch a Jupyter Binder with a ready-to-use virtual environment and boilerplate code.

Use the following binder for quantile-unbiased analysis.

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gl/dsbowen%2Fconditional-inference/HEAD?urlpath=lab/tree/examples/rqu.ipynb

| Use the following binder for Bayesian analysis.

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gl/dsbowen%2Fconditional-inference/HEAD?urlpath=lab/tree/examples/bayes.ipynb

|
Installation
============

.. code-block::

   $ pip install conditional-inference

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

   @software(bowen2021conditional-inference,
      title={ Conditional Inference },
      author={ Bowen, Dillon },
      year={ 2021 },
      url={ https://dsbowen.gitlab.io/conditional-inference }
   )

   @techreport{andrews2019inference,
      title={ Inference on winners },
      author={ Andrews, Isaiah and Kitagawa, Toru and McCloskey, Adam },
      year={ 2019 },
      institution={ National Bureau of Economic Research }
   }

Acknowledgements
================

I would like to thank Isaiah Andrews, Toru Kitagawa, Adam McCloskey, and Jeff Rowley for invaluable feedback on my early drafts.