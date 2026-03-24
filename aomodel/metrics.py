import numpy as np
import aomodel.utils as utils
import aomodel.estimation as estimation

# Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2026-1309.

def slopes_tps(data_values, locations=None, axis=2, time_block_size=None, sampling_frequency=None, remove_mean=True,
               use_overlapping_blocks=True):
    """
    Takes the temporal power spectrum (TPS) of the slopes (i.e., gradient) of input data with respect to a specified
    axis. This function i) approximates the slopes of the data (using a second order finite difference method) with
    respect to the specified axis and ii) calls ``temporal_power_spectrum`` to estimate the TPS of these slopes.

    Args:
        data_values (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing the
            data values.
        locations (Union[int, ndarray], optional): [Default=None] either a numpy 1-D array containing the coordinates of
            the axis to which the gradient is to be calculated or an integer specifying a uniform distance between
            bins. Sent as an argument to ``numpy.gradient``.

            - If set to None, this argument is not included in the call to ``numpy.gradient``.

        axis (int, optional): [Default=2] the axis of data values to take the gradient with respect to.

            - If axis = 2, the gradient is calculated with respect to the x-axis.

            - If axis = 1, the gradient is calculated with respect to the y-axis.

            - If axis = 0, the gradient is calculated with respect to time.

        time_block_size (int, optional): [Default=None] the size of each time block to use for the TPS estimation.
            The full time-series is broken up into "time-blocks" of the indicated size. For each time-block, the TPS is
            calculated independently. The final TPS calculation is then the average over each time-block. This value
            must be a positive integer and can be at most the number of time-steps in data_values.

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
    """
    assert (axis in [0, 1, 2])

    if locations is None:
        data_slopes = np.gradient(data_values, axis=axis)
    else:
        data_slopes = np.gradient(data_values, locations, axis=axis)

    # Sets the mask to be the intersection of valid data values for all images in the time-series:
    mask = (np.average(1 - np.uint8(np.isnan(data_slopes)), axis=0) == 1)
    return temporal_power_spectrum(data_values=data_slopes, time_block_size=time_block_size, mask=mask,
                                   sampling_frequency=sampling_frequency, remove_mean=remove_mean,
                                   use_overlapping_blocks=use_overlapping_blocks)


def temporal_power_spectrum(data_values, time_block_size=None, mask=None, sampling_frequency=None, remove_mean=True,
                            use_overlapping_blocks=True):
    """
    Uses the ``vector_temporal_psd`` function from :ref:`Estimation` to estimate the temporal power spectrum (TPS) of
    an input time series of 2-D arrays (i.e., images).

    Args:
        data_values (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing the
            data values

        time_block_size (int, optional): [Default=None] the size of each time block to use for the TPS estimation.
            The full time-series is broken up into "time-blocks" of the indicated size. For each time-block, the TPS is
            calculated independently. The final TPS calculation is then the average over each time-block. This value
            must be a positive integer and can be at most the number of time-steps in data_values.

            - If set to None, the function finds the block size so that the TPS estimates are averaged over at least 100
              blocks. This ensures that the signal-to-noise ratio of each 1-D TPS estimate is at least 10.

        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width)
            indicating which 2-D indices of each time-step correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

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
    """
    assert (len(data_values.shape) == 3)

    if mask is None:
        # Sets the mask to be the intersection of valid data values for all images in the time-series:
        mask = (np.average(1 - np.uint8(np.isnan(data_values)), axis=0) == 1)
    else:
        # Checks the mask on all images in the data set:
        assert (mask.dtype == bool)
        assert (mask.shape == (data_values.shape[1], data_values.shape[2]))
        assert (not np.isnan(utils.img_to_vec(image_data=data_values, mask=mask)).any())

    # Converts the input OPD images to vectors:
    data_vector = utils.img_to_vec(image_data=data_values, mask=mask)

    return estimation.vector_temporal_power_spectrum(data_values=data_vector, time_block_size=time_block_size,
                                                     sampling_frequency=sampling_frequency, remove_mean=remove_mean,
                                                     use_overlapping_blocks=use_overlapping_blocks)


def structure_function_2d(data, mask=None, compute_square_root=False):
    """
    Estimate the 2-D spatial structure function of an input time-series of 2-D data. As described in
    :cite:`UtleyBoiling, UtleyBoiling2`, the 2-D structure function is defined as a function of a two-dimensional
    separation vector, instead of a one-dimensional separation distance. This function first normalizes the data by its
    sample mean and standard deviation (which is called the quasi-homogeneous structure matrix in :cite:`Vogel`) and
    then estimates the structure function.

    Args:
        data (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing the data from
            which we would like to estimate the 2-D structure function.

        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width)
            indicating which 2-D indices of each time-step correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

        compute_square_root (bool, optional): [Default=False] choice of whether to compute (and return) the square root
            of the structure function values (instead of the structure function values themselves).

    Returns:
        - **structure_function_inputs** (*ndarray*) -- numpy 2-D array of shape (number of pixel differences, 2)
          containing the 2-D inputs to the structure function (in polar coordinates).
        - **structure_function** (*ndarray*) -- numpy 1-D array of length (number of pixel differences) containing the
          estimated structure function values (in the same order as the first output).
    """
    if mask is None:
        # Sets the mask to be the intersection of valid data values for all images in the time-series:
        mask = (np.average(1 - np.uint8(np.isnan(data)), axis=0) == 1)

    # Normalize the data's statistics by removing the moving and dividing by the standard deviation
    # (along each pixel):
    data_flat = utils.img_to_vec(image_data=data, mask=mask).T
    mean = np.average(data_flat, axis=1)
    data_mean_removed = data_flat - mean[:, np.newaxis]
    standard_deviation_vector = np.sqrt(np.sum(data_mean_removed ** 2, axis=1) / data.shape[0])

    # If there are (temporally) constant components, allow normalization:
    with np.errstate(divide='ignore', invalid='ignore'):
        data_normalized = data_mean_removed / standard_deviation_vector[:, np.newaxis]

    # Calculate the list of relative separations between pixels (units: number of pixels)
    spatial_indices = np.argwhere(mask)
    differences = spatial_indices[:, np.newaxis, :] - spatial_indices[np.newaxis, :, :]
    relative_separations = np.linalg.norm(differences, axis=2)
    # Extract unique relative separations
    relative_separations = relative_separations[np.triu_indices(n=relative_separations.shape[0], k=1)]

    # Compute the quasi-homogeneous spatial structure function using the normalized data
    spatial_covariance = (1.0 / data.shape[0]) * (data_normalized @ data_normalized.T)
    # Extract the spatial covariance values that we need for structure function calculations
    covariance_values = spatial_covariance[np.triu_indices(n=spatial_covariance.shape[0], k=1)]
    structure_function_values = 2 * (1 - covariance_values)

    # Take the square root of all structure function values (if prompted to by the user):
    if compute_square_root:
        structure_function_values = np.sqrt(structure_function_values)

    # Sort the relative separation values in ascending order
    sort_indices = np.argsort(relative_separations)
    relative_separations = relative_separations[sort_indices]

    # Sort the structure function array accordining to the same indices
    structure_function_values = structure_function_values[sort_indices]

    # Compute and sort the angle of each difference
    angles = np.arctan2(differences[:, :, 0], differences[:, :, 1])
    angles = angles[np.triu_indices(n=angles.shape[0], k=1)]
    angles = np.mod(angles[sort_indices], np.pi)

    # Average the structure function values of each (relative separation, angle) pair
    unique_relative_separations = np.unique(relative_separations)
    structure_function_inputs = []
    structure_function = []
    for index, relative_separation in enumerate(unique_relative_separations):
        relative_separation_indices = np.squeeze(np.argwhere(relative_separations == relative_separation))
        associated_angles = angles[relative_separation_indices]
        unique_associated_angles = np.sort(np.unique(associated_angles))
        for angle in unique_associated_angles:
            angle_indices = np.squeeze(np.argwhere(angles == angle))
            intersect_indices = np.intersect1d(relative_separation_indices, angle_indices)
            structure_function_inputs.append([relative_separation, angle])
            structure_function.append(np.average(structure_function_values[intersect_indices]))

    structure_function_inputs = np.array(structure_function_inputs)
    structure_function = np.array(structure_function)

    return structure_function_inputs, structure_function


