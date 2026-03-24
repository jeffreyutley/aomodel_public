.. _PCA:

===
PCA
===

Implements Principal Component Analysis (PCA) to fit the spatial statistics of a time series of images. Used by
the :ref:`ReVAR` and :ref:`LongRangeAR` modules.

This module can be used to both i) compute PCA from data and ii) generate synthetic data images with the same spatial
statistics as input data.

Functions
---------

.. automodule:: aomodel.pca
    :members: find_top_principal_components, compute_pca, generative_pca_algorithm
    :undoc-members:
    :show-inheritance:

    .. rubric:: **Functions:**

    .. autosummary::
       find_top_principal_components
       compute_pca
       generative_pca_algorithm