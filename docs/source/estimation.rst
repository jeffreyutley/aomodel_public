.. _Estimation:

==========
Estimation
==========

Functions used by :ref:`LongRangeAR` to estimate parameters from training data.

* ``estimate_long_range_ar_parameters`` estimates and returns all parameters of a ``LongRangeAR`` instance.
* ``vector_temporal_power_spectrum`` and ``compute_low_pass_filter_params`` are used to compute the low-pass
  filter parameters.
* ``least_squares_solution`` computes the prediction weights of a ``LongRangeAR`` model.

Functions
---------

.. automodule:: aomodel.estimation
    :members: estimate_long_range_ar_parameters, compute_low_pass_filter_params, least_squares_solution, vector_temporal_power_spectrum
    :undoc-members:
    :show-inheritance:

    .. rubric:: **Functions:**

    .. autosummary::
       estimate_long_range_ar_parameters
       compute_low_pass_filter_params
       least_squares_solution
       vector_temporal_power_spectrum