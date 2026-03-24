.. _InstallationDocs:

============
Installation 
============

The ``AOMODEL`` package currently is only available to download and install from source through GitHub.


**Downloading and installing from source**

1. Download the source code:

In order to download the python code, move to a directory of your choice and run the following two commands::

    git clone https://github.com/jeffreyutley/aomodel_public.git
    cd aomodel

2. Install the conda environment and package:

It is recommended that you install to a virtual environment. There are two options to do so:

a. *Clean install from dev_scripts*

*******You can skip all other steps if you do a clean install.******

To do a clean install, use the command::

    cd dev_scripts
    source clean_install_all.sh


b. *Manual install*

If you have Anaconda installed, you can run the following::

    conda env create -f environment.yml
    conda activate aomodel

This creates a conda environment ``aomodel`` using the ``environment.yml`` file. Now to use the package, this
``aomodel`` environment needs to be activated.

3. Verify Installation:

You can verify the installation by running ``pip show aomodel``, which should display a brief summary of the
packages installed in the ``aomodel`` environment. Now you will be able to use the ``aomodel`` python commands from
any directory by running the python command ``import aomodel``.

