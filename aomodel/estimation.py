import numpy as np
import aomodel.utils as utils
import aomodel.pca as pca
import aomodel._indexing as indexing


def estimate_long_range_ar_parameters(training_data, time_lags, prediction_window_indices, predicted_components,
                                      num_low_pass_filters, low_pass_filter_params=None, cutoff_frequency=None,
                                      tps_block_size=None):
    """
    Estimate the parameters used by an instance of the ``LongRangeAR`` class to generate synthetic data.

    Args:
        training_data (ndarray): numpy 2-D array of shape (data vector dimensionality, number of time samples)
            containing the data to fit.

        time_lags (ndarray): numpy 1-D array of time lags to use for the model.

        prediction_window_indices (ndarray): numpy 2-D integer-valued array of shape
            (number of predicted vector components, maximum prediction window size) containing the indices for which we
            calculate prediction weights.

            - In the case that one vector component has a smaller prediction window size
              than others, the remaining spaces in the row associated to the component should be filled in with -1 (to
              the right of the true prediction window indices).

        predicted_components (ndarray): numpy 1-D integer array containing the indices of the data vector components for
            which the function will compute prediction weights.

        num_low_pass_filters (int, optional): [Default=0] the number of low-pass filters to use in the linear time
            predictive model.

        low_pass_filter_params (ndarray, optional): [Default=None] numpy 1-D array of length ``num_low_pass_filters``
            containing the parameters of each low-pass filter.

            - If set to None or has length zero, the function assumes that no low-pass filters are used.

        cutoff_frequency (float, optional): [Default=None] cutoff frequency (in units of cycles/sample) to use for
            estimation of the low-pass filter parameters.

            - If set to None and low-pass filters are used, then this function estimates the cutoff frequency as the
              peak frequency of the temporal power spectrum (TPS) of the training data.

        tps_block_size (int, optional): [Default=None] time-block size for calculating the TPS of the training data.

            - If set to None and the low-pass filter parameters need to be estimated, finds the block size so that the
              TPS estimates are averaged over 100 blocks. This ensures that the signal-to-noise ratio of each 1-D TPS
              estimate is at least 10.

    Returns:
        **parameters** (*dict*) -- a dictionary containing the estimated model parameters with the following keys

        - **prediction_weights** (*ndarray*) -- numpy 1-D array containing the (concatenated) prediction weights for
          each vector component.
        - **low_pass_filter_params** (*ndarray*) -- numpy 1-D array of length ``num_low_pass_filters`` containing the
          parameters of each low-pass filter.
        - **residuals_mean** (*ndarray*) -- numpy 1-D array of length ``vector_dimensionality`` containing the mean
          vector of the residuals of the least-squares fit.
        - **noise_modulation** (*ndarray*) -- numpy 2-D array of shape
          (``vector_dimensionality``, ``vector_dimensionality``) scaling the (unit-variance) white noise to match the
          covariance of the residuals.
    """
    assert (num_low_pass_filters >= 0)
    if low_pass_filter_params is None:
        low_pass_filter_params = np.zeros(num_low_pass_filters)

    assert (len(low_pass_filter_params) == num_low_pass_filters)

    # If low-pass filters are used, estimates the low-pass filter parameters using a cutoff frequency:
    if (low_pass_filter_params == 0).all():
        if num_low_pass_filters > 0:
            # If not provided, estimates the cutoff frequency as the peak frequency of the TPS of the training data:
            if cutoff_frequency is None:
                # Computes the TPS of only the vector components which are included in the prediction window:
                num_components = len(predicted_components)
                truncated_data = training_data[:num_components]     # Top components of the data

                # Computes the TPS:
                frequencies, tps = vector_temporal_power_spectrum(
                    data_values=truncated_data.T, time_block_size=tps_block_size
                )

                # Defines the cut-off frequency as the peak frequency of the TPS:
                cutoff_frequency = utils.parabolic_interpolation_max(frequencies, tps)

            # Computes the low-pass filter parameters from the cut-off frequency:
            low_pass_filter_params = compute_low_pass_filter_params(
                initial_cutoff_frequency=cutoff_frequency,
                num_low_pass_filters=num_low_pass_filters
            )

    # Uses least squares regression to estimate the prediction weights for each vector component:
    prediction_weights, residuals = least_squares_solution(data=training_data,
                                                           time_lags=time_lags,
                                                           prediction_window_indices=prediction_window_indices,
                                                           predicted_components=predicted_components,
                                                           low_pass_filter_params=low_pass_filter_params)

    # Computes a spatial PCA from the residuals:
    residuals_mean, residuals_principal_components, residuals_singular_values = pca.compute_pca(residuals)

    # Calculates the noise modulation matrix (which scales unit-variance white noise to map onto the spatial
    # distribution of the residuals) from the PCA:
    noise_modulation = np.multiply(residuals_principal_components, np.sqrt(residuals_singular_values)[None, :])

    # Return a dictionary of all learned parameters
    parameters = {
        "prediction_weights": prediction_weights,
        "low_pass_filter_params": low_pass_filter_params,
        "residuals_mean": residuals_mean,
        "noise_modulation": noise_modulation
    }
    return parameters


def compute_low_pass_filter_params(initial_cutoff_frequency, num_low_pass_filters):
    """
    Calculates the parameters of a set of low-pass filters given an initial cut-off frequency. For the i-th low-pass
    filter, the cut-off frequency is i orders of magnitude below the initial cut-off frequency.

    Args:
        initial_cutoff_frequency (float): initial cut-off frequency (in [cycles per time-step]) for the low-pass
            filters.
        num_low_pass_filters (int): number of low-pass filters to create.

    Returns:
        **low_pass_filter_params** (*ndarray*) -- numpy 1-D array of length ``num_low_pass_filters`` containing the
        parameters of each low-pass filter.
    """
    assert ((initial_cutoff_frequency > 0) and (initial_cutoff_frequency <= 0.5))

    low_pass_filter_params = np.zeros(num_low_pass_filters)
    low_pass_filter_params[0] = (2 * np.pi * initial_cutoff_frequency) / 10.0

    # Sets additional low-pass filter parameters to an order of magnitude below the first parameter:
    for low_pass_filter_index in range(1, num_low_pass_filters):
        low_pass_filter_params[low_pass_filter_index] = low_pass_filter_params[low_pass_filter_index - 1] / 10.0

    return low_pass_filter_params


def least_squares_solution(data, time_lags, prediction_window_indices, predicted_components,
                           low_pass_filter_params=None):
    """
    Uses least-squares regression to estimate the prediction weights for a **Long-Range AR** model, as described in
    :ref:`LongRangeAR`. This function takes into account the "prediction window", which determines which weights to
    estimate and which to leave as zero. The (non-zero) estimated prediction weights are output in a 1-D array, in the
    order of the vector components to which they are associated. This solution is partially adapted from
    :cite:`Lutkepohl`, which derives the solution for a Vector AR model.

    Args:
        data (ndarray): numpy 2-D array of shape (data vector dimensionality, number of samples) containing the
            ground-truth data values for the regression calculation.

        time_lags (ndarray): numpy 1-D array of time lags to use for the model.

        prediction_window_indices (ndarray): numpy 2-D integer-valued array of shape
            (number of predicted vector components, maximum prediction window size) containing the indices for which to
            calculate prediction weights.

            - In the case that one vector component has a smaller prediction window size
              than others, the remaining spaces in the row associated to the component should be filled in with -1 (to
              the right of the true prediction window indices).

        predicted_components (ndarray): numpy 1-D integer array containing the indices of the data vector components for
            which the function will compute prediction weights.

        low_pass_filter_params (ndarray, optional): [Default=None] numpy 1-D array of length ``num_low_pass_filters``
            containing the parameters of each low-pass filter.

            - If set to None, the function assumes that no low-pass filters are used.

    Returns:
        - **prediction_weights_array** (*ndarray*) -- numpy 1-D array containing the (concatenated) prediction
          weights for each vector component.
        - **residuals** (*ndarray*) -- numpy 2-D array of shape
          (vector dimensionality, ``num_samples`` - ``max(time_lags)``) containing the residuals of the linear
          prediction.
    """
    assert (time_lags.ndim == 1)
    assert np.issubdtype(time_lags.dtype, np.integer)
    assert (time_lags > 0).all()

    assert np.issubdtype(predicted_components.dtype, np.integer)
    assert (data.ndim == prediction_window_indices.ndim == 2)

    assert (predicted_components.ndim == 1)
    assert np.issubdtype(predicted_components.dtype, np.integer)
    assert ((0 <= predicted_components).all() and (predicted_components < data.shape[0]).all())

    if low_pass_filter_params is None:
        num_low_pass_filters = 0
    else:
        num_low_pass_filters = len(low_pass_filter_params)

    vector_dimensionality = data.shape[0]
    num_samples = data.shape[1]

    # Array that stores the linear prediction (from the previous values multiplied by the prediction weights):
    prediction = np.zeros((vector_dimensionality, num_samples - max(time_lags)))

    # Creates indexing arrays for the compressed prediction window:
    valid_prediction_window_indices = (prediction_window_indices != -1)
    data_time_step_indices_for_regression = np.arange(num_samples - max(time_lags))[:, np.newaxis]
    time_step_offsets_for_prediction_window, vector_component_indices_for_prediction_window, lpf_column_index_map = (
        indexing.prediction_window_indexing_arrays(vector_dimensionality=vector_dimensionality,
                                                   time_lags=time_lags,
                                                   prediction_window_indices=prediction_window_indices,
                                                   num_low_pass_filters=num_low_pass_filters))
    num_lpf_columns = len(lpf_column_index_map)

    # Place the low-pass filter parameters into arrays:
    if num_lpf_columns > 0:
        low_pass_input_contribution_weights = low_pass_filter_params[lpf_column_index_map]
        low_pass_memory_retention_weights = 1.0 - low_pass_input_contribution_weights

    # Uses a 1-D array to store the prediction weights:
    prediction_weights_array = np.array([])

    # Use least squares regression to calculate prediction weights independently for each vector component:
    for i, comp in enumerate(predicted_components):
        # Fill in prediction array for current vector component "comp":
        vector_component_indices = \
            np.tile(vector_component_indices_for_prediction_window[i, valid_prediction_window_indices[i]],
                    (num_samples - max(time_lags), 1))
        time_step_indices = np.tile(max(time_lags) +
                                    time_step_offsets_for_prediction_window[i, valid_prediction_window_indices[i]],
                                    (num_samples - max(time_lags), 1)) + data_time_step_indices_for_regression

        # Defines the prediction array containing the previous (known) data values:
        if num_lpf_columns > 0:
            # Array containing the previous data vectors (excluding low-pass filters):
            data_vector_arrays = (
                data[vector_component_indices[:, num_lpf_columns:], time_step_indices[:, num_lpf_columns:]])

            # Extract data to use as input to low-pass filter calculation:
            lpf_vector_component_indices = vector_component_indices_for_prediction_window[i, :num_lpf_columns]
            lpf_input_data = data[lpf_vector_component_indices, :].T

            # Compute low-pass filters and store in a single 2-D array
            low_pass_filters = np.zeros_like(lpf_input_data)
            low_pass_filters[1] = low_pass_input_contribution_weights * lpf_input_data[0]
            for time_index in range(2, num_samples):
                low_pass_filters[time_index] = (low_pass_memory_retention_weights * low_pass_filters[time_index - 1] +
                                                low_pass_input_contribution_weights * lpf_input_data[time_index - 1])

            # Full prediction array (all values to prediction weights for):
            prediction_array = np.concatenate((low_pass_filters[max(time_lags):], data_vector_arrays), axis=1)
        else:
            # If there are no low-pass filters, the prediction array is just the previous data values:
            prediction_array = data[vector_component_indices, time_step_indices]

        # Ground-truth data values:
        component_data = data[comp, max(time_lags):num_samples, np.newaxis]

        # Estimates the MMSE prediction weights:
        scaling = np.dot(prediction_array.T, prediction_array)

        # Calculate least squares solution:
        if np.linalg.matrix_rank(scaling) == scaling.shape[0]:
            prediction_weights = np.linalg.solve(scaling, np.dot(prediction_array.T, component_data))
        else:
            # If the scaling matrix is not full rank, uses the lstsq function from numpy.linalg:
            prediction_weights = np.linalg.lstsq(a=prediction_array, b=component_data, rcond=None)[0]

        # Append new prediction weights to the 1-D array:
        prediction_weights_array = np.concatenate((prediction_weights_array, np.squeeze(prediction_weights, axis=1)))

        # Calculates linear prediction using the new prediction weights:
        prediction[comp] = np.squeeze(np.dot(prediction_array, prediction_weights))

    # Compute the residuals of the linear prediction:
    residuals = data[:, max(time_lags):num_samples] - prediction

    return prediction_weights_array, residuals


def vector_temporal_power_spectrum(data_values, time_block_size=None, sampling_frequency=None, remove_mean=True,
                                   use_overlapping_blocks=True):
    """
    Estimates the Temporal Power Spectrum (TPS) of an input time series of vectors by averaging the 1-D TPS estimates
    for each vector component (i.e., for each time series of data values at a single vector component). Each 1-D TPS is
    estimated using Welch's method :cite:`Welch`, in which the time series is broken up into independent "blocks" of
    length ``time_block_size``. A Hamming window is applied to each block and the TPS is estimated using an FFT and a
    scaling factor which normalizes the variance of the estimate. This function is partially adapted from
    :cite:`Poyneer, Oppenheim` and is described in detail in :cite:`UtleyBoiling2`.

    Args:
        data_values (ndarray): numpy 2-D array of shape (number of time-steps, vector dimensionality) containing the
            data values. This function requires at least 1,000 time-steps of data.

        time_block_size (int, optional): [Default=None] the size of each time block to use for the TPS estimation.
            This value must be a positive integer and can be at most the number of time-steps in data_values.

            - If set to None, the function finds the block size so that the TPS estimates are averaged over at least 100
              blocks. This ensures that the signal-to-noise ratio of each 1-D TPS estimate is at least 10.

        sampling_frequency (float, optional): [Default=None] the temporal sampling frequency of the discrete-time signal
            ``data_values``. This input should be included if the desired TPS units are energy per unit time instead of
            energy per unit sample. In this case, the frequency bins are in units of cycles per unit time instead of
            cycles per unit sample.

            - If set to None, the TPS units are energy/sample and the frequency units are cycles/sample.

        remove_mean (bool, optional): [Default=True] choice of removing the temporal mean of each vector component
            before computing the TPS. It is recommended to set this to True in most cases. If set to False, the lowest
            frequencies may be offset.

        use_overlapping_blocks (bool, optional): [Default=True] whether to use overlapping time-blocks when calculating
            the TPS. If set to True, then the time-blocks will have a 50% overlap. This method allows one to maintain
            the same block size while also reducing noise in the TPS calculation. If set to False, then the time-blocks
            will have no overlap.

    Returns:
        - **frequencies** (*ndarray*) -- numpy 1-D array containing the frequency bins of the TPS calculation.
        - **tps_estimate** (*ndarray*) -- numpy 1-D array containing the TPS estimates for each frequency bin.

    Raises:
        ValueError: if the input data_values has fewer than 1,000 time-steps.
    """
    assert (data_values.ndim == 2)

    if data_values.shape[0] < 1_000:
        raise ValueError(f"The input data_values must have at least 1,000 time-steps, got {data_values.shape[0]}.")

    if time_block_size is not None:
        assert (0 < time_block_size <= data_values.shape[0])
    else:
        # Makes sure we average over 100 time-blocks:
        if use_overlapping_blocks:
            time_block_size = int(data_values.shape[0] // 50.5)
        else:
            time_block_size = int(data_values.shape[0] // 100.0)
        # Make sure time block size is even
        if time_block_size % 2 == 1:
            time_block_size -= 1

    # Calculates the number of time-blocks we average over:
    if use_overlapping_blocks:
        num_blocks = 2 * (data_values.shape[0] // time_block_size) - 1
    else:
        num_blocks = data_values.shape[0] // time_block_size

    # Initializes a Hamming window and computes relevant quantities:
    hamming_window = np.hamming(time_block_size)
    welch_scaling_factor = np.sum(hamming_window ** 2)

    # Frequencies along the x-axis:
    frequencies = np.fft.rfftfreq(n=time_block_size)

    # Removes the temporal mean from each vector component:
    if remove_mean:
        temporal_mean = np.mean(data_values, axis=0)
        data_values = data_values - temporal_mean[np.newaxis, :]

    # Iterates over each time-block and averages the TPS:
    tps_estimates = np.zeros((len(frequencies), data_values.shape[1]))
    for block_idx in range(num_blocks):
        # Extracts the time-series for the current block:
        if use_overlapping_blocks:
            block_data = data_values[int((block_idx / 2) * time_block_size):
                                     int(((block_idx / 2) + 1) * time_block_size)]
        else:
            block_data = data_values[block_idx * time_block_size: (block_idx + 1) * time_block_size]

        # If mean-removal is not selected, takes the FFT of the windowed block data (with its mean):
        windowed_data = block_data * hamming_window[:, np.newaxis]

        # Uses an FFT to approximate the DFT of the current section:
        block_dft = np.fft.rfft(windowed_data, axis=0)

        # Uses the FFT to approximate the energy spectrum of the current time-block:
        block_energy_spectrum = np.abs(block_dft) ** 2

        # Applies the Welch's method scaling factor to compute the TPS estimate for this block
        tps_estimates += block_energy_spectrum / welch_scaling_factor

    # Averages the TPS estimates across each row (i.e., across the estimates for each pixel) and divides by the number
    # of sections used:
    tps_estimate = np.average(tps_estimates, axis=1) / num_blocks

    # If a sampling frequency is provided, divides the TPS estimate by the sampling frequency to ensure correct unit
    # conversion:
    if sampling_frequency:
        assert (sampling_frequency > 0)
        frequencies = sampling_frequency * frequencies
        tps_estimate = tps_estimate / sampling_frequency

    return frequencies, tps_estimate
