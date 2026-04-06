.. aomodel documentation master file, created by
   sphinx-quickstart on Fri Jun 25 14:24:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AOMODEL
=======

**AOMODEL** is a Python package for generating synthetic aero-optic phase screen data.

The main feature of this package is an implementation of the **Re-whitened Vector AutoRegression (ReVAR)** algorithm,
which generates synthetic time series of images that match the statistics of measured data.

This package can be used to:

* Fit the statistics of input data.
* Generate synthetic data with the desired statistics.
* Evaluate the statistics of an input data set.

For more information about the algorithm, see :cite:`Utley2, Utley`.

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Background

   theory
   advanced_features
   references

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: User Guide

   install
   api_overview
   api
   demo