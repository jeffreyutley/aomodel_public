.. _Utilities:

=========
Utilities
=========

Utilities used by the ``aomodel`` package.

* ``vec_to_img`` and ``img_to_vec`` convert quickly from 2-D arrays (images) to 1-D vectors (rasterized images). These
  functions are used in :ref:`ReVAR` and :ref:`Metrics`.
* ``parabolic_interpolation_max`` is used in :ref:`LongRangeAR` when finding a cut-off frequency.

Functions
---------

.. automodule:: aomodel.utils
    :members: vec_to_img, img_to_vec, parabolic_interpolation_max
    :undoc-members:
    :show-inheritance:

    .. rubric:: **Functions:**

    .. autosummary::
       vec_to_img
       img_to_vec
       parabolic_interpolation_max