.. _Theory:

======
Theory
======

ReVAR
=====

The following describes how the **Re-whitened Vector AutoRegression (ReVAR)** algorithm works. **ReVAR** takes an input
time series of images and uses three steps to fit the spatial and temporal correlations of this training data:

1.  **Pre-processing**: Reduce the spatial dimensionality of the training data using a spatial Principal Component
    Analysis (PCA).
2.  **Long-Range AutoRegression**: Capture temporal correlations within the training data by training a linear
    predictive model on the dimensionality-reduced data. Take the residuals of this linear predictive model.
3.  **Re-whitening**: Capture spatial correlations within the training data by taking the residuals of the linear
    predictive model and computing a second spatial PCA. This converts the residuals to *white noise*.

These steps estimate parameters which embed the spatial and temporal correlations of the training data.

``aomodel`` uses two classes to implement this algorithm: ``LongRangeAR`` and ``ReVAR``. ``ReVAR`` implements the
pre-processing step and then uses ``LongRangeAR`` to implement the next two steps. See :ref:`AdvancedFeatures` for
additional ways to use ``LongRangeAR`` without defaulting to the conventions of **ReVAR**.

Pre-Processing
--------------

Pre-processing first computes a spatial PCA by taking the Singular Value Decomposition (SVD) of the data's spatial
covariance matrix:

.. math::
    R_X = E \Lambda E^T.

We then use this PCA to represent the training data in the basis of principal components,

.. math::
    \mathbf{\tilde{X}}_n = E^T \mathbf{X}_n,

and find a *prediction subspace* containing the 99% of the spatial variance. The training data represented in the
prediction subspace is denoted as :math:`\mathbf{X}_n^{(P)}`.

This reduction in the spatial dimensionality of the training data lowers the computational expense and parameter count
of the **Long-Range AR** step. This allows us to increase the numbers of time-lags and low-pass filters without
overfitting to the training data.

Long-Range AR
-------------

The **Long-Range AR** model is designed to generate synthetic vector time-series that match the statistical properties
of a training dataset. It addresses a common limitation of standard Auto-Regressive (AR) models: the inability to
capture both short-range and long-range temporal correlations simultaneously without exploding the parameter count.

**Long-Range AR** employs a hybrid architecture combining two components:

1.  **Short-Range AR:** Captures high-frequency temporal correlations.
2.  **Long-Range Low-Pass Filters (LPF):** Captures low-frequency temporal correlations.

**ReVAR** applies this model to the prediction subspace representation of the training data: :math:`\mathbf{X}_n^{(P)}`.

Linear Prediction
~~~~~~~~~~~~~~~~~

The core of **Long-Range AR** is a linear predictive model. For a data vector :math:`\mathbf{\tilde{X}}^{(P)}_n` at
time-step :math:`n`, the prediction is formulated as:

.. math::

    \mathbf{\hat{X}}_n = \sum_{\ell=1}^{N_L} A_{X,\ell} \mathbf{\tilde{X}}^{(P)}_{n-\ell} + \sum_{i=1}^{N_F} A_{Y,i}
    \mathbf{Y}_{i,n-1}.

Where:

* :math:`A_{X,\ell}` are the *prediction weight* matrices for the standard time-lags (determined by ``time_lags``).

* :math:`\mathbf{Y}_{i,n}` is the state of the :math:`i`-th low-pass filter of the data
  :math:`\mathbf{\tilde{X}}^{(P)}_n`, which captures long-range temporal correlations.

* :math:`A_{Y,i}` are the prediction weights applied to the low-pass filters :math:`\mathbf{Y}_{i,n}`.

Low-Pass Filtering (The "Long Range" Component)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard AR models require thousands of lag coefficients to "remember" events from the distant past. **Long-Range AR**
instead uses recursive *Low-Pass Filters (LPFs)* to summarize this history efficiently.

If ``num_low_pass_filters > 0`` in the ``ReVAR`` or ``LongRangeAR`` classes, the model maintains a recursive state:

.. math::

    \mathbf{Y}_{i,n} = (1 - \alpha_i) \cdot \mathbf{Y}_{i,n-1} + \alpha_i \cdot \mathbf{\tilde{X}}^{(P)}_{n}

The parameter :math:`\alpha_i` is determined by a cut-off frequency of the low-pass filter :math:`\mathbf{Y}_{i,n}`.
This allows the model to retain memory over long horizons using only a single extra prediction weight matrix
:math:`A_{Y,i}` per filter, rather than thousands of additional matrices :math:`A_{X,\ell}` applied to standard
time-lags.

Re-whitening
------------

After fitting the temporal correlations, the model calculates the residuals:

.. math::

    \boldsymbol{\xi}_n = \mathbf{\tilde{X}}_n - \mathbf{\hat{X}}_n.

These residuals are temporally un-correlated, but spatially correlated. In fact, the spatial correlations of the
residuals embed the spatial correlations of the data :math:`\mathbf{X}_n` itself. To capture these spatial correlations,
we perform a second spatial PCA on :math:`\boldsymbol{\xi}_n`:

.. math::
    R_{\boldsymbol{\xi}} = U \Sigma U^T.

The matrices :math:`U` and :math:`\Sigma` allow us to convert the residuals to **white noise**:

.. math::
    \boldsymbol{W}_n = \Sigma^{-1/2} U^T \boldsymbol{\xi}_n.

Synthetic Data Generation
-------------------------

To generate synthetic data, **ReVAR** takes white noise input and inverts the above three steps. This results in the
following procedure:

1.  **Spatial-correlating**: Convert the white noise to spatially-correlated noise using the second spatial PCA.
2.  **Long-Range AR Synthesis**: Convert the spatially-correlated noise to temporally-correlated data by applying
    the linear predictive model.
3.  **Post-processing**: Convert this temporally-correlated data to synthetic data using the first spatial PCA.

Applying these steps using a pre-trained model will generate synthetic time series of images with i) the same image size
as the training data and ii) arbitrarily-many time-steps.

Theoretical Assumptions
-----------------------

**ReVAR** assumes that the training data follows a multivariate Gaussian distribution whose statistics of temporally
stationary (i.e., invariant to time-shifts). Synthetic data generated by the algorithm will follow this distribution by
default.

If the training data does not follow this assumed theoretical model, then **ReVAR** may not accurately fit the
statistics of the data.