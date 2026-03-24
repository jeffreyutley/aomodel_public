.. _Demo:

====
Demo
====

The ``demo/`` directory contains two Python scripts: ``demo_1_data_generation.py`` and
``demo_2_parameter_estimation.py``.

The file ``demo_1_data_generation.py`` demonstrates how to apply the :ref:`ReVAR` module to i) generate synthetic data
with a pre-trained model and ii) compare statistics of the synthetic data to those of measured aero-optic phase screen
data.

The file ``demo_2_parameter_estimation.py`` demonstrates how to estimate the parameters of ReVAR from the measured
aero-optic phase screen data.

Before running this script, see :ref:`DemoDataDownloading` for instructions on downloading the measured data sets used
in this demos.

After downloading the data files, run the demo scripts from the parent directory with something like the following
command::

    python demo/demo_file.py

* :ref:`DemoDataDownloading`
* :ref:`DemoUtilities`

.. _demo_data_downloading: demo_data_downloading.html
.. _demo_utilities: demo_utilities.html

.. toctree::
   :titlesonly:
   :hidden:

   demo_data_downloading
   demo_utilities