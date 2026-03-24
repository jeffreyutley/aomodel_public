.. docs-include-ref

AOMODEL
=======

This project includes a data-driven algorithm that generates synthetic time-series of images (of arbitrary duration)
by estimating statistical parameters from an input time-series of images.

..
    Include more detailed description here.

Installing
----------
1. Clone or download the repository:

    .. code-block::

        git clone git@github.com:jeffreyutley/aomodel_public

2. Install the conda environment and package

    a. Option 1: Clean install from dev_scripts

        *******You can skip all other steps if you do a clean install.******

        To do a clean install, use the command:

        .. code-block::

            cd dev_scripts
            source clean_install_all.sh

    b. Option 2: Manual install

        Create a conda environment ``aomodel`` using the ``environment.yml`` file.

        .. code-block::

            conda env create -f environment.yml

        Anytime you want to use this package, this ``aomodel`` environment should be activated with the following:

        .. code-block::

            conda activate aomodel


Running Demo(s)
---------------

The demo scripts ``demo_1_data_generation.py`` and ``demo_2_parameter_estimation.py`` show examples of how to use the
ReVAR algorithm to i) generate synthetic data that matches the statistics of measured data sets and ii) estimate the
parameters of ReVAR from measured data.

Before running the demo script, download the measured data sets:

    Option 1. Install using shell script

        Use the script ``get_demo_data_server.sh`` inside of the ``demo`` folder to automatically install the data and
        place it in the proper folder for the scripts ``demo_1_data_generation.py`` and
        ``demo_2_parameter_estimation.py``.

        Inside of the parent directory (the aomodel_public folder containing this file), run the following:

        .. code-block::

            source demo/get_demo_data_server.sh

    Option 2. Manual install

        To manually install the data sets, visit the
        `Bouman data repository <https://www.datadepot.rcac.purdue.edu/bouman/>` and download the .zip file
        ``TBL_data.zip``.

        Unzip the file and place the folder ``TBL_data`` inside of the ``data/demo`` directory.

Run either of the demo scripts from the parent directory (the aomodel_public folder containing this file) with something
like the following command:

    .. code-block::


        python demo/demo_file.py