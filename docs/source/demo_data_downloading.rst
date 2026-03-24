.. _DemoDataDownloading:

==========================
Downloading Demo Data Sets
==========================

The demo scripts ``demo_1_data_generation.py`` and ``demo_2_parameter_estimation.py`` use two measured data sets that
must first be downloaded.

**Option 1. Install using shell script**

Use the script ``get_demo_data_server.sh`` inside of the ``demo`` folder to automatically install the data and
place it in the proper folder for the scripts ``demo_1_data_generation.py`` and ``demo_2_parameter_estimation.py``.

Inside of the parent directory (the aomodel_public folder containing README.rst), run the following::

    source demo/get_demo_data_server.sh

**Option 2. Manual install**

To manually install the data sets, visit the Bouman-data-repository_ and download the .zip file ``TBL_data.zip``.

Unzip the file and place the folder ``TBL_data`` inside of the ``data/demo`` directory.

.. _Bouman-data-repository: https://www.datadepot.rcac.purdue.edu/bouman/