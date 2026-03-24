.. _Metrics:

=======
Metrics
=======

The ``Metrics`` module contains functions which compute the statistics of either measured or synthetic
data. Comparison of these statistics between measured and synthetic data can be used to evaluate the accuracy
of the methods used in ``ReVAR``, ``Long-RangeAR``, and ``PCA``..

* ``slopes_tps`` and ``temporal_power_spectrum`` evaluate the temporal statistics of data.
* ``structure_function_2d`` evaluate spatial statistics of data.

.. automodule:: aomodel.metrics
    :members: slopes_tps, temporal_power_spectrum, structure_function_2d
    :undoc-members:
    :show-inheritance:

    .. rubric:: **Functions:**

    .. autosummary::
       slopes_tps
       temporal_power_spectrum
       structure_function_2d