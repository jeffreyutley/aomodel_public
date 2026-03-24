.. _ReVAR:

=====
ReVAR
=====

The ``ReVAR`` class implements **ReVAR (Re-whitened Vector AutoRegression)**, which combines the :ref:`PCA`
and :ref:`LongRangeAR` modules to generate synthetic time series of images with the same spatial and temporal statistics
as input data.

This implementation is object-oriented, with the ``ReVAR`` class used to both i) fit the model to measured data and ii)
generate synthetic data.

Constructor
-----------
.. autoclass:: aomodel.ReVAR
    :show-inheritance:

Model Fitting
-------------
.. automethod:: aomodel.ReVAR.fit
.. automethod:: aomodel.ReVAR.pre_processing

Saving and Loading Models
-------------------------
.. automethod:: aomodel.ReVAR.save
.. automethod:: aomodel.ReVAR.load

Generating Synthetic Data
-------------------------
.. automethod:: aomodel.ReVAR.run

Properties
----------
.. autoproperty:: aomodel.ReVAR.num_parameters