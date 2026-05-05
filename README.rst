.. docs-include-ref

AOMODEL
=======

This project includes a data-driven algorithm that generates synthetic time-series of images (of arbitrary duration)
by estimating statistical parameters from an input time-series of images. Full documentation is available at
https://aomodel-public.readthedocs.io .

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

Before running either demo script, download the measured data sets. Before running ``demo_1_data_generation.py``, you
must also download pre-trained models. There are two options to download the data:

    Option 1. Install using shell script

        Use the script ``get_demo_data_server.sh`` inside of the ``demo`` folder to automatically install both the data
        and the pre-trained models. This script also places the files in the proper folders for the scripts
        ``demo_1_data_generation.py`` and ``demo_2_parameter_estimation.py``.

        Inside of the parent directory (the aomodel_public folder containing this file), run the following:

        .. code-block::

            source demo/get_demo_data_server.sh

    Option 2. Manual install

        To manually install the data sets and pre-trained models, visit the
        `Bouman data repository <https://www.datadepot.rcac.purdue.edu/bouman/>` and download the .zip files
        ``TBL_data.zip`` and ``pre_trained_models.zip`` (respectively).

        Unzip the two files, then place the folder ``TBL_data`` inside of the ``demo/data`` directory and the files
        ``F06_pre_trained_model.npz`` and ``F12_pre_trained_model.npz`` inside of the ``demo/pre_trained_models``
        directory.

Run either of the demo scripts from the parent directory (the aomodel_public folder containing this file) with something
like the following command:

    .. code-block::


        python demo/demo_file.py


Disclaimer
----------

Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2026-1309.