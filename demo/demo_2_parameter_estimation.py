import aomodel
import numpy as np
import os

# Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2026-1309.

"""
This file demonstrates how to estimate the parameters of ReVAR from input data. We show an example with two measured
turbulent boundary layer (TBL) data sets F06 and F12. The result of this code is a pre-trained model (saved to file)
for both TBL data sets.

Although the other demo file demo_1_data_generation.py uses pre-trained models to generate data, it could be slightly
modified to instead used the models saved by this file.

For this example, we have specified four time-lags and two low-pass filters for the Long-Range AR model. These values
can be manually adjusted for each data set to modify the resolution of the temporal statistics captured by ReVAR.
Increasing the number of time-lags will increase resolution at the high frequencies, while increasing the number of 
low-pass filters will increase resolution at the low frequencies. Although increasing both of these parameters provides
greater temporal resolution, they will also increase the memory requirements and computation time of both parameter
estimation and data generation.

Another input that can be adjusted is the percent variance included in the prediction subspace of the Long-Range AR
model. This demo uses the ReVAR algorithm, which fixes this value at 99%. However, the ReVAR class within the aomodel
package allows the the user to specify this input if desired. Increasing the percent variance will provide the 
Long-Range AR model with additional low-variance principal components, but can significantly increase the memory
requirements and computation time of the algorithm.

Before running this file, see the docs for instructions on downloading these TBL data sets.
"""

datasets = ['F06', 'F12']

# Number of time-lags and low-pass filters to use for the Long-Range AR model:
num_time_lags = 4
num_low_pass_filters = 2

# Percentage of the total spatial variance to include in the prediction subspace:
percent_variance = 0.99

# File name of the ReVAR parameters based on the previous input values
save_filename = (f'ReVAR_parameters_'
                 f'{num_time_lags}_time_lags_'
                 f'{num_low_pass_filters}_low_pass_filters_'
                 f'{percent_variance}_percent_variance.npz')

for dataset in datasets:
    print("Data Set %s" % dataset + '\n'
          "============\n"
          "============\n")

    # Build filepaths
    load_filepath = f'./demo/data/TBL_data/TBL_data_set_{dataset}.npz'
    save_filepath = f'./demo/output/data_set_{dataset}_' + save_filename

    print(f"-- Loading TBL_data_set_{dataset}.npz ---")
    print(f"Looking for file at: {os.path.abspath(load_filepath)}")

    # Load .npz data file
    try:
        file = np.load(load_filepath)
        print("File loaded successfully.")
    except FileNotFoundError:
        print("❌ ERROR: File not found!")
        print("Check that the TBL_data folder was downloaded and placed correctly.")
        raise
    except Exception as e:
        print("❌ ERROR while loading the .npz file:")
        print(e)
        raise

    # Extract arrays
    print("Extracting arrays from file...")

    phase_screens = file['phase_screens']           # [radians]
    mask = file['mask']
    wavelength = file['wavelength'] * 1e6           # [microns]

    print(f"--- Done loading TBL_data_set_{dataset}.npz ---\n")

    # Convert phase to Optical Path Difference (OPD)
    measured_data = phase_screens * wavelength / (2 * np.pi)  # [microns]

    # Extract the training data from the TBL data
    num_training_time_steps = int(0.8 * measured_data.shape[0])
    training_data = measured_data[:num_training_time_steps]

    # Construct the model:
    model = aomodel.ReVAR(time_lags=num_time_lags,
                          data_mask=mask,
                          num_low_pass_filters=num_low_pass_filters)

    # Estimate the parameters of ReVAR from the training data:
    model.fit(training_data=training_data,
              percent_variance=percent_variance)

    # Save the parameters to file so that it can be used to generate synthetic data:
    model.save(save_filepath)

    print('\n')
