.. _DemoUtilities:

==============
Demo Utilities
==============

The following functions are used as utilities by the demo file ``demo_1_data_generation.py``.

* ``create_video``, ``structure_function_image_array``, ``plot_structure_function_image``, and ``plot_tps`` are used to
  create visualizations of the data images or statistics of the data.
* ``compute_rms`` and ``compute_nrmse`` compute scalar metrics by taking some type of difference between statistics of
  input data and those of synthetic data.

.. automodule:: demo.demo_utils
    :members: create_video, structure_function_image_array, plot_structure_function_image, plot_tps, compute_rms, compute_nrmse
    :undoc-members:
    :show-inheritance:

    .. rubric:: **Functions:**

    .. autosummary::
       create_video
       structure_function_image_array
       plot_structure_function_image
       plot_tps
       compute_rms
       compute_nrmse