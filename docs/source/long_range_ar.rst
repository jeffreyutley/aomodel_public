.. _LongRangeAR:

=========================
Long-Range AutoRegression
=========================

The ``Long-Range AutoRegression`` class implements a **Long-Range AR** model to generate synthetic time series of
vectors with the same short-range and long-range temporal statistics as input data.

This implementation is object-oriented, with the ``LongRangeAR`` class used to both i) fit the model to measured
data and ii) generate synthetic data.

Constructor
------------
.. autoclass:: aomodel.LongRangeAR
    :show-inheritance:

Model Fitting
-------------
.. automethod:: aomodel.LongRangeAR.fit

Saving and Loading Models
-------------------------
.. automethod:: aomodel.LongRangeAR.save
.. automethod:: aomodel.LongRangeAR.load

Generating Synthetic Data
-------------------------
.. automethod:: aomodel.LongRangeAR.run

Defining Model Parameters
-------------------------
.. automethod:: aomodel.LongRangeAR.create_model_structure

Properties
----------
.. autoproperty:: aomodel.LongRangeAR.prediction_window_mask
.. autoproperty:: aomodel.LongRangeAR.predicted_components
.. autoproperty:: aomodel.LongRangeAR.remaining_components
.. autoproperty:: aomodel.LongRangeAR.num_low_pass_filters
.. autoproperty:: aomodel.LongRangeAR.num_prediction_weights
.. autoproperty:: aomodel.LongRangeAR.num_parameters