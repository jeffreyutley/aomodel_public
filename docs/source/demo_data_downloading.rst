.. _DemoDataDownloading:

==========================
Downloading Demo Data Sets
==========================

The demo scripts ``demo_1_data_generation.py`` and ``demo_2_parameter_estimation.py`` use two measured data sets that
must first be downloaded. Further, the script ``demo_1_data_generation.py`` uses a pre-trained model for each data set
that must also be downloaded.

**Option 1. Install using shell script**

Use the script ``get_demo_data_server.sh`` inside of the ``demo`` folder to automatically install both the data and
pre-trained models and then place them in the proper folders for the scripts ``demo_1_data_generation.py`` and
``demo_2_parameter_estimation.py``.

Inside of the parent directory (the aomodel_public folder containing README.rst), run the following::

    source demo/get_demo_data_server.sh

**Option 2. Manual install**

To manually install the data sets, visit the Bouman-data-repository_ and download the .zip files ``TBL_data.zip`` and
``pre_trained_models.zip``.

Unzip the two files, then place the folder ``TBL_data`` inside of the ``demo/data`` directory and the files
``F06_pre_trained_model.npz`` and ``F12_pre_trained_model.npz`` inside of the ``demo/pre_trained_models`` directory.

.. _Bouman-data-repository: https://www.datadepot.rcac.purdue.edu/bouman/