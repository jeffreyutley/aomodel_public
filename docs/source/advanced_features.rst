.. _AdvancedFeatures:

Advanced Features
=================

The **ReVAR** algorithm by default trains the Long-Range AR model on a *prediction subspace* spanning the top principal
coefficients of the training data. This reduces the spatial dimensionality of the data and thus reduces the number of
prediction weights in the **Long-Range AR** model, which can be necessary to prevent overfitting. For this reason, the
``ReVAR`` class first uses the method ``pre_processing`` to find this prediction subspace and uses
``LongRangeAR`` with this specific prediction structure.

However, this prediction subspace is far from the only way to reduce the number of prediction weights in
**Long-Range AR**. For many data sets, other options may also be effective.

For this reason, the ``LongRangeAR`` class is flexible to different prediction structures. That is, you can specify
which vector components to use for predicting every other vector component. We call this choice of spatial prediction
structure the **Prediction Window**.

This section covers how to use built-in prediction window structures, define your own custom structure, and discover
structure automatically.

Built-in Prediction Window Structures
-------------------------------------
The package provides standard prediction window structures for common data types. These are passed to the model via the
``prediction_window_structure`` argument.

**1. Distance-Based Structure**
Use this for data with a spatial or logical order (e.g., sensors along a line). It enforces a "neighbor" relationship,
assuming periodic boundaries (circular wrapping).

.. code-block:: python

    from aomodel.prediction_window_structure import DistanceWindowStructure

    # Each component is predicted by neighbors within distance 1
    structure = DistanceWindowStructure(distance=1)

**2. Prediction Subspace**
This is the default structure used by ``ReVAR``. Use this for hierarchical data (like PCA modes). It models connections
only among the top vector components.

.. code-block:: python

    from aomodel.prediction_window_structure import SubspaceWindowStructure

    # Only the first 5 components predict each other
    structure = SubspaceWindowStructure(subspace_dimension=5)

**3. Full-Window Structure**
If the data has a low spatial dimensionality, the linear predictive model will likely benefit from including all
possible vector component pairs inside of the prediction window.

.. code-block:: python

    from aomodel.prediction_window_structure import FullWindowStructure

    # Include all possible vector component pairs
    structure = FullWindowStructure()

In the case that you don't want certain vector components to be predicted (e.g., low-variance PCA modes), you can
specify which vector components are assigned prediction weights. Vector components which are not assigned prediction
weights will be fully modeled by the spatial distribution of the residuals :math:`\boldsymbol{\xi}_n`.

.. code-block:: python

    from aomodel.prediction_window_structure import FullWindowStructure
    import numpy as np

    vector_dimensionality = 10                  # Spatial dimension of the data is 10.
    predicted_components = np.arange(0, 5)      # Only predict the top 5 vector components.

    # Use all 10 components to predict the top 5
    structure = FullWindowStructure(predicted_components=predicted_components)


Defining Custom Structures
--------------------------
If your data has a complex topology (e.g., a specific sensor grid that isn't linear), you can define a custom prediction
mask.

**Option A: Explicit Boolean Mask**
You can pass a boolean matrix directly using ``ExplicitWindowStructure``. The mask must have shape
``(Vector Dimensionality, Vector Dimensionality)``. A ``True`` value at ``mask[i, j]`` indicates that component ``j``
is used to predict component ``i``. In this case, a non-zero prediction weight is assigned to this index.

.. code-block:: python

    import numpy as np
    from aomodel.prediction_window_structure import ExplicitWindowStructure

    # Create a custom mask for 3 components
    mask = np.array([
        [True,  False, True],  # Comp 0 predicted by 0 and 2
        [False, True,  False], # Comp 1 predicted by 1 only
        [True,  True,  True]   # Comp 2 predicted by everyone
    ])

    model = LongRangeAR(
        vector_dimensionality=3,
        prediction_window_structure=ExplicitWindowStructure(mask)
    )

**Option B: Subclassing**
For reusable logic, you can subclass ``PredictionWindowStructure`` and implement the ``get_mask`` method.

.. code-block:: python

    from aomodel.prediction_window_structure import PredictionWindowStructure

    class MyCustomGrid(PredictionWindowStructure):
        def get_mask(self, vector_dimensionality):
            mask = ... # Your logic here
            return mask

