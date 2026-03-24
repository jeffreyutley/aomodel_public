import aomodel
import numpy as np
import demo_utils
import os

# Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2026-1309.

"""
This file demonstrates how to use the ReVAR algorithm to generate synthetic data with the same statistics of measured
data sets. This code i) generates synthetic data using a pre-trained ReVAR model and ii) compares the statistics of 
the synthetic data with those of the measured data. For the measured data, this file uses two turbulence boundary layer
(TBL) data sets F06 and F12.

The pre-trained models used here were trained on these two TBL data sets. For both F06 and F12, the pre-trained models
used four time-lags and two low-pass filters, which we found empirically to be optimal for these two data sets.

To use a different ReVAR model (e.g., with different time-lags and low-pass filters), the input parameters_filepath can
be adjusted to load a different model. The demo file demo_2_parameter_estimation.py demonstrates how to compute and
save a pre-trained model to a specific file-path. Although this demo file also uses four time-lags and two low-pass 
filters, these values are easily adjusted in the demo file.

The statistics computed by this code (to compare the measured and synthetic data) are the Temporal Power Spectrum (TPS)
applied to both the data itself and the spatial finite difference of the data in the x-direction, the 2-D structure
function of the data, and the root-mean square (RMS) of the data. The TPS evaluates temporal statistics, the structure
function evaluates spatial statistics, and the RMS evaluates the overall spatial-temporal fit.

Before running this file, see the docs for instructions on downloading these TBL data sets and pre-trained model 
files.
"""

datasets = ['F06', 'F12']

# Number of time-lags and low-pass filters used by the pre-trained model:
num_time_lags = 4
num_low_pass_filters = 2

# Block sizes for the Temporal Power Spectrum (TPS) calculations:
opd_tps_block_sizes = {'F06': 596, 'F12': 994}
flow_tps_block_sizes = {'F06': 298, 'F12': 496}

# Parameters of plotting functions
video_colorbar_scales = {'F06': 0.875, 'F12': 0.8}
video_figsize = {'F06': (15, 5), 'F12': (10, 5)}
video_fontsize = {'F06': 40, 'F12': 30}
structure_function_colorbar_scales = {'F06': 0.4, 'F12': 0.525}

for dataset in datasets:
    print("Data Set %s" % dataset + '\n'
          "============\n"
          "============\n")

    # Build filepaths
    data_filepath = f'./demo/data/TBL_data/TBL_data_set_{dataset}.npz'  # Measured data
    parameters_filepath = f'./demo/pre_trained_models/{dataset}_pre_trained_model.npz'   # ReVAR parameters

    print(f"-- Loading TBL_data_set_{dataset}.npz ---")
    print(f"Looking for file at: {os.path.abspath(data_filepath)}")

    # Load .npz data file
    try:
        file = np.load(data_filepath)
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
    pixel_spacing = file['pixel_spacing'] * 1e6     # [microns]
    sampling_frequency = file['sampling_frequency'] # [Hz]
    wavelength = file['wavelength'] * 1e6           # [microns]

    print(f"--- Done loading TBL_data_set_{dataset}.npz ---\n")

    # Convert phase to Optical Path Difference (OPD)
    measured_data = phase_screens * wavelength / (2 * np.pi)  # [microns]

    # Extract the training and testing data from the TBL data
    num_training_time_steps = int(0.8 * measured_data.shape[0])
    num_synthetic_time_steps = measured_data.shape[0] - num_training_time_steps

    # Block sizes for TPS calculations:
    opd_tps_block_size = opd_tps_block_sizes[dataset]
    flow_tps_block_size = flow_tps_block_sizes[dataset]

    # Calculates the RMS and flow and OPD TPS for the measured data:
    measured_opd_rms = demo_utils.compute_rms(data_values=measured_data,
                                              mask=mask)
    frequencies_opd_tps, measured_opd_tps = (
        aomodel.metrics.temporal_power_spectrum(data_values=measured_data[num_training_time_steps:],
                                                time_block_size=opd_tps_block_size,
                                                sampling_frequency=sampling_frequency,
                                                remove_mean=True))
    frequencies_flow_tps, measured_flow_tps = (
        aomodel.metrics.slopes_tps(data_values=measured_data[num_training_time_steps:],
                                   locations=pixel_spacing,
                                   axis=2,
                                   time_block_size=flow_tps_block_size,
                                   sampling_frequency=sampling_frequency,
                                   remove_mean=True))

    # Finds the spatial structure function values of the OPD data:
    structure_function_data = (
        aomodel.metrics.structure_function_2d(data=measured_data[num_training_time_steps:],
                                              mask=mask))
    structure_function_inputs, measured_structure_function = structure_function_data
    measured_structure_function_sqrt = (
        aomodel.metrics.structure_function_2d(data=measured_data[num_training_time_steps:],
                                              mask=mask,
                                              compute_square_root=True)[1])

    # Creates an instance of the class ReVAR and loads pre-trained model:
    model = aomodel.ReVAR(time_lags=num_time_lags,
                          data_mask=mask,
                          num_low_pass_filters=num_low_pass_filters,
                          load_file=parameters_filepath)

    # Generates synthetic data using this model:
    synthetic_data = model.run(num_time_steps=num_synthetic_time_steps)

    # Calculates the RMS and TPS of both OPD and deflection angle for the (synthetic) data:
    estimated_opd_rms = demo_utils.compute_rms(data_values=synthetic_data,
                                               mask=mask)
    estimated_opd_tps = aomodel.metrics.temporal_power_spectrum(data_values=synthetic_data,
                                                                time_block_size=opd_tps_block_size,
                                                                sampling_frequency=sampling_frequency,
                                                                remove_mean=True)[1]
    estimated_flow_tps = aomodel.metrics.slopes_tps(data_values=synthetic_data,
                                                    locations=pixel_spacing,
                                                    axis=2,
                                                    time_block_size=flow_tps_block_size,
                                                    sampling_frequency=sampling_frequency,
                                                    remove_mean=True)[1]

    # Finds the spatial structure function values of the synthetic data:
    estimated_structure_function = aomodel.metrics.structure_function_2d(data=synthetic_data,
                                                                         mask=mask)[1]
    estimated_structure_function_sqrt = aomodel.metrics.structure_function_2d(data=synthetic_data,
                                                                              mask=mask,
                                                                              compute_square_root=True)[1]

    # Creates a video of the (first 100 frames of the) data alongside synthetic data:
    measured_data_video = measured_data[num_training_time_steps:num_training_time_steps + 100, 2:-2, 2:-2]
    synthetic_data_video = synthetic_data[:100, 2:-2, 2:-2]
    vid = demo_utils.create_video(data=measured_data_video,
                                  title='Measured Data',
                                  mask=mask[2:-2, 2:-2],
                                  data2=synthetic_data_video,
                                  title2='Synthetic Data',
                                  figsize=video_figsize[dataset],
                                  cbar_scale=video_colorbar_scales[dataset],
                                  fontsize=video_fontsize[dataset])
    vid.save(f'./demo/output/data_set_{dataset}_video.gif')

    # Plots the TPS of both the OPD values and the synthetic data:
    demo_utils.plot_tps(frequencies=frequencies_opd_tps / 1_000,    # convert from [Hz] to [kHz]
                        tps_values=measured_opd_tps,
                        tps_values_2=estimated_opd_tps,
                        x_label='Frequency $f$ [kHz]',
                        y_label='TPS Value $S_{OPD}(f)$',
                        title=f'Data Set {dataset}: $OPD$ Temporal Power Spectrum',
                        label1='Measured Data',
                        label2='Synthetic Data',
                        savefile=f'./demo/output/data_set_{dataset}_OPD_TPS.png')

    # Plots the TPS of the deflection angle for both the OPD values and the synthetic data:
    demo_utils.plot_tps(frequencies=frequencies_flow_tps / 1_000,   # convert from [Hz] to [kHz]
                        tps_values=measured_flow_tps,
                        tps_values_2=estimated_flow_tps,
                        x_label='Frequency $f$ [kHz]',
                        y_label='TPS Value $S_{\\theta_x}(f)$',
                        title=f'Data Set {dataset}: Flow Temporal Power Spectrum',
                        label1='Measured Data',
                        label2='Synthetic Data',
                        savefile=f'./demo/output/data_set_{dataset}_flow_TPS.png')

    # Creates images of both structure functions:
    demo_utils.plot_structure_function_image(
        structure_function_inputs=structure_function_inputs,
        structure_function=measured_structure_function,
        structure_function_2=estimated_structure_function,
        image_title='Measured Data',
        image_title_2='Synthetic Data',
        suptitle=f'Data Set {dataset}: Structure Function',
        cbar_scale=structure_function_colorbar_scales[dataset],
        savefile=f'./demo/output/data_set_{dataset}_measured_structure_function.png')

    # Scalar metrics
    phase_tps_error = demo_utils.compute_nrmse(ground_truth_data=measured_opd_tps,
                                               estimated_data=estimated_opd_tps)
    flow_tps_error = demo_utils.compute_nrmse(ground_truth_data=measured_flow_tps,
                                              estimated_data=estimated_flow_tps)
    structure_function_error = demo_utils.compute_nrmse(ground_truth_data=measured_structure_function_sqrt,
                                                        estimated_data=estimated_structure_function_sqrt)
    opd_rms_error = np.abs(measured_opd_rms - estimated_opd_rms) / measured_opd_rms

    print("\n================================================")
    print(f"     Measured Data Set {dataset}: Scalar Metric Values")
    print("================================================")

    print(f"{'OPD TPS Error:':35s} {phase_tps_error}")
    print(f"{'Flow TPS Error:':35s} {flow_tps_error}")
    print(f"{'Structure Function Error:':35s} {structure_function_error}")
    print(f"{'OPD_rms Error:':35s} {opd_rms_error}")

    print("\n\n")
