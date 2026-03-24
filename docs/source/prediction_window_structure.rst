.. _PredictionWindowStructure:

===========================
Prediction Window Structure
===========================

This module defines an abstract base class ``PredictionWindowStructure``, which is used as a template for creating the
prediction window mask used in :ref:`LongRangeAR`.

The following classes derive from this base class: ``FullWindowStructure``, ``SubspaceWindowStructure``,
``DistanceWindowStructure``, and ``ExplicitWindowStructure``. Each of these classes creates a prediction window mask
based on some structure.

For more details on using these classes, see :ref:`AdvancedFeatures`.

These classes are documented below:

Abstract Base Class
-------------------

.. autoclass:: aomodel.prediction_window_structure.PredictionWindowStructure
    :show-inheritance:
.. automethod:: aomodel.prediction_window_structure.PredictionWindowStructure.get_mask()

Inherited Classes
-----------------

The inherited classes each have the ``get_mask`` method derived from ``PredictionWindowStructure``. In each inherited
class, the method takes in the same argument ``vector_dimensionality`` and returns the same array ``mask``.

.. autoclass:: aomodel.prediction_window_structure.FullWindowStructure
    :show-inheritance:

.. autoclass:: aomodel.prediction_window_structure.SubspaceWindowStructure
    :show-inheritance:

.. autoclass:: aomodel.prediction_window_structure.DistanceWindowStructure
    :show-inheritance:

.. autoclass:: aomodel.prediction_window_structure.ExplicitWindowStructure
    :show-inheritance: