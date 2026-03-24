import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import aomodel
import scipy

# Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2026-1309.

def create_video(data, title='', mask=None, data2=None, title2='', figsize=None, cbar_scale=1.0, fontsize=None):
    """
    Creates a video from a sequence of images using ``matplotlib.pyplot`` and ``matplotlib.animation``. Shows each image
    on the same scale and includes a colorbar. Each frame of the video can either have a single image or two images
    side-by-side.

    Args:
        data (ndarray): numpy 3-D array of shape (number of frames, image height, image width) containing the data
            values to create a video of.

        title (str, optional): [Default=''] title to place above the video.

        mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width)
            indicating which 2-D indices of each time-step correspond to valid pixel values.

            - If set to None, the function infers the mask based on which data values are "nan."

        data2 (ndarray, optional): [Default=None] numpy 3-D array with the same shape as ``data`` and which contains
            additional data to create a video from. If included, each frame of the video will include two images
            side-by-side.

            - If set to None, only one image is included in each frame.

        title2 (str, optional): [Default=''] title above the images from ``data2``. Only used by the function if
            ``data2`` is included as an input argument.

        figsize (tuple of float, optional): [Default=None] length-2 tuple containing the (width, height) of the figure,
            where both width and height are floating-point values. Units are in inches.
            
            - If set to None, the default figsize in ``matplotlib.pyplot`` is used.
            
        cbar_scale (float, optional): [Default=1.0] the scale to shrink the color-bar.

    Returns:
        **video** (*matplotlib.animation.ArtistAnimation*) -- the ``ArtistAnimation`` figure containing the video.
    """
    assert (len(data.shape) == 3)

    if mask is None:
        # Sets the mask to be the intersection of valid data values for all images in the sequence:
        mask = (np.average(1 - np.uint8(np.isnan(data)), axis=0) == 1)
    else:
        # Checks the mask on all images in the data set:
        assert (mask.dtype == bool)
        assert (mask.shape == (data.shape[1], data.shape[2]))
        assert (not np.isnan(aomodel.utils.img_to_vec(image_data=data, mask=mask)).any())

    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]

    if fontsize is None:
        fontsize = plt.rcParams["font.size"]

    num_frames = data.shape[0]
    video_frames = []

    if data2 is not None:
        # Ensures that the mask is valid for data2:
        assert (not np.isnan(aomodel.utils.img_to_vec(image_data=data2, mask=mask)).any())

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # Finds the minimum and maximum values of the data for the purpose of colorbar creation:
        datavals = aomodel.utils.img_to_vec(image_data=data, mask=mask)
        data2vals = aomodel.utils.img_to_vec(image_data=data2, mask=mask)
        vmin = np.array([datavals.min(), data2vals.min()]).min()
        vmax = np.array([datavals.max(), data2vals.max()]).max()

        # Adding each frame to the list:
        for idx in range(num_frames):
            img = axs[0].imshow(data[idx], interpolation='none', vmin=vmin, vmax=vmax, animated=True)
            img2 = axs[1].imshow(data2[idx], interpolation='none', vmin=vmin, vmax=vmax, animated=True)
            video_frames.append([img, axs[0].set_title(title, fontsize=fontsize, loc='center', pad=15), img2,
                                 axs[1].set_title(title2, fontsize=fontsize, loc='center', pad=15)])

        axs[0].axis('off')
        axs[1].axis('off')

    else:
        fig = plt.figure(figsize=figsize)
        axs = plt.axes()
        datavals = aomodel.utils.img_to_vec(image_data=data, mask=mask)
        vmin = datavals.min()
        vmax = datavals.max()

        # Adding each frame to the list:
        for idx in range(num_frames):
            img = plt.imshow(data[idx], interpolation='none', vmin=vmin, vmax=vmax, animated=True)
            video_frames.append([img, plt.title(title, fontsize=fontsize, loc='center', pad=15)])

        axs.axis('off')

    # Add the colorbar:
    cbar = fig.colorbar(img, ax=axs, fraction=0.046, pad=0.04, shrink=cbar_scale)
    cbar.ax.tick_params(labelsize=20, pad=10)

    video = ani.ArtistAnimation(fig, video_frames)

    return video


def structure_function_image_array(structure_function_inputs, structure_function, interpolation_scale=2,
                                   interpolate_using_zero=True):
    """
    Uses ``matplotlib.pyplot`` to create an image of the 2-D structure function. The structure function is plotted in
    rectangular coordinates and is extended the structure function to the lower halfplane. The function interpolates at
    the center of each pixel using bi-linear interpolation (according to ``interpolation_scale``).

    Args:
        structure_function_inputs (ndarray): numpy 2-D array of shape (number of pixel differences, 2)
          containing the 2-D inputs to the structure function (in polar coordinates).

        structure_function (ndarray): numpy 1-D array of length (number of pixel differences) containing the
          estimated structure function values (in the same order as the first output).

        interpolation_scale (int, optional): [Default=2] the (integer) value used to scale the number of pixels along
            each axis.

        interpolate_using_zero (bool, optional): [Default=True] whether to use a value of zero at the origin to
            interpolate at the center of each pixel.

    Returns:
        - **structure_function_image** (*ndarray*) -- numpy 2-D array containing the structure function values to plot
          as an image (at each pixel).
        - **image_extent** (*tuple of float*) -- length-4 tuple containing the input to the ``extent`` parameter of
          ``matplotlib.pyplot.imshow``.
    """

    # Extend angles from the range [0, pi) to [0, 2*pi):
    additive_pi = np.tile(np.array([0, np.pi]), (structure_function_inputs.shape[0], 1))
    additional_structure_function_inputs = structure_function_inputs + additive_pi
    extended_structure_function_inputs = np.zeros((2 * structure_function_inputs.shape[0], 2))
    extended_structure_function = np.zeros(2 * structure_function.shape[0])
    extended_structure_function_inputs[:structure_function_inputs.shape[0]] = structure_function_inputs
    extended_structure_function_inputs[structure_function_inputs.shape[0]:] = additional_structure_function_inputs
    extended_structure_function[:structure_function_inputs.shape[0]] = structure_function
    extended_structure_function[structure_function_inputs.shape[0]:] = structure_function

    # Convert from polar coordinates (r, theta) to rectangular coordinates (x, y):
    num_points = extended_structure_function_inputs.shape[0]
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    for point_index in range(num_points):
        r = extended_structure_function_inputs[point_index, 0]
        theta = extended_structure_function_inputs[point_index, 1]
        x[point_index], y[point_index] = (r * np.cos(theta), r * np.sin(theta))

    # Sets the pixel lengths along the x- and y-axes:
    x_pixel_length, y_pixel_length = 1 / interpolation_scale, 1 / interpolation_scale

    # Sets the number of numbers of pixels along each axis:
    num_x_pixels = interpolation_scale * int(x.max()-x.min()+1)
    num_y_pixels = interpolation_scale * int(y.max()-y.min()+1)

    # Create pixel centers
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_values = np.linspace(x_min + 0.5*x_pixel_length, x_max - 0.5*x_pixel_length, num_x_pixels)
    y_values = np.linspace(y_min + 0.5*y_pixel_length, y_max - 0.5*y_pixel_length, num_y_pixels)

    # If "interpolate_using_zero" is True, add the origin and a value of zero to the list of structure function values:
    if interpolate_using_zero:
        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)
        extended_structure_function = np.insert(extended_structure_function, 0, 0)

    # Create the structure function image by interpolating (bi-linear) at the center of each pixel:
    X, Y = np.meshgrid(x_values, y_values)
    structure_function_image = scipy.interpolate.griddata((x, y), extended_structure_function, (X, Y),
                                                          method='linear')

    # Removes the pixels closest to zero from the image (by setting the values to "nan"):
    lowest_distance = np.min(np.sqrt(X ** 2 + Y ** 2))
    close_to_zero_indices = np.argwhere(np.abs(np.sqrt(X ** 2 + Y ** 2) - lowest_distance) < 1e-10)
    for i in range(close_to_zero_indices.shape[0]):
        zero_index = (close_to_zero_indices[i][0], close_to_zero_indices[i][1])
        structure_function_image[zero_index] = np.nan

    image_extent = (x_values[0] - 0.5*x_pixel_length, x_values[-1] + 0.5*x_pixel_length,
                    y_values[0] - 0.5*y_pixel_length, y_values[-1] + 0.5*y_pixel_length)

    return structure_function_image, image_extent


def plot_structure_function_image(structure_function_inputs, structure_function, structure_function_2=None,
                                  image_title='', image_title_2='', suptitle='', figsize=None, cbar_scale=1.0,
                                  savefile=None, show=True):
    """
    Uses ``matplotlib.pyplot`` to create an image of the 2-D structure function. Can either create a single structure
    function image or two images side-by-side (according to the same scale and colorbar).

    Args:
        structure_function_inputs (ndarray): numpy 2-D array of shape (number of pixel differences, 2)
          containing the 2-D inputs to the structure function (in polar coordinates).

        structure_function (ndarray): numpy 1-D array of length (number of pixel differences) containing the
          estimated structure function values (in the same order as the first output).

        structure_function_2 (ndarray): numpy 1-D array of the same shape as ``structure_function`` containing a second
            structure function.

        image_title (str, optional): [Default=''] the title of the image.

        image_title_2 (str, optional): [Default=''] the title of the second image.

        suptitle (str, optional): [Default=''] the suptitle of the image.

        figsize (tuple of float, optional): [Default=None] length-2 tuple containing the (width, height) of the figure,
            where both width and height are floating-point values. Units are in inches.

            - If set to None, the default figsize in matplotlib.pyplot is used.

        cbar_scale (float, optional): [Default=1.0] the scale to shrink the color-bar.

        savefile (str, optional): [Default=None] the filename to save the figure to.

            - If set to None, the figure is not saved.

        show (bool, optional): [Default=True] whether to display the image.

    Returns:
        **fig** (*matplotlib.figure*) -- the figure created by the function.
    """
    assert (structure_function.ndim == 1)
    assert ((structure_function_inputs.ndim == 2) and (structure_function_inputs.shape[1] == 2))

    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]

    # Form the structure function image array
    structure_function_image, image_extent = structure_function_image_array(structure_function_inputs,
                                                                            structure_function)

    # Create image figure:
    if structure_function_2 is not None:
        assert (structure_function_2.shape == structure_function.shape)
        structure_function_image_2 = structure_function_image_array(structure_function_inputs, structure_function_2)[0]
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
        im = axs[0].imshow(structure_function_image, interpolation='none', aspect='equal', origin='lower',
                           extent=image_extent)
        im = axs[1].imshow(structure_function_image_2, interpolation='none', aspect='equal', origin='lower',
                           extent=image_extent)
        fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04, shrink=cbar_scale)
        axs[0].set_title(image_title, fontsize=15)
        axs[1].set_title(image_title_2, fontsize=15)
        plt.suptitle(suptitle, fontsize=20)

    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        im = ax.imshow(structure_function_image, interpolation='none', aspect='equal', origin='lower',
                       extent=image_extent)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=cbar_scale)
        plt.title(image_title, loc='center', fontsize=15)

    if savefile is not None:
        plt.savefig(savefile)

    if show:
        plt.show()

    return fig


def plot_tps(frequencies, tps_values, tps_values_2=None, x_label='', y_label='', title='', log_scale=False,
             label1='Input Data', label2='Synthetic Data', savefile=None, show=True):
    """
    Uses ``matplotlib.pyplot`` to plot the Temporal Power Spectrum (TPS) of measured and/or synthetic data. This
    function can plot a single array of values or two arrays in the same graph.

    Args:
        frequencies (ndarray): numpy 1-D array containing the frequency bins of the TPS values.

        tps_values (ndarray): numpy 1-D array containing the TPS values at each frequency bin.

        tps_values_2 (ndarray): numpy 1-D array containing a second set of TPS values at each frequency bin.

        x_label (str, optional): [Default=''] the label to display along the x-axis.

        y_label (str, optional): [Default=''] the label to display along the y-axis.

        title (str, optional): [Default=''] the title of the plot.

        log_scale (bool, optional): [Default=False] whether to use a logarithmic scale on the x-axis.

        label1 (str, optional): [Default=''] the label of the ``tps_values`` plot.

        label2 (str, optional): [Default=''] the label of the ``tps_values_2`` plot.

        savefile (str, optional): [Default=None] the filename to save the figure to.

            - If set to None, the figure is not saved.

        show (bool, optional): [Default=True] whether to display the image.

    Returns:
        **fig** (*matplotlib.figure*) -- the figure created by the function.
    """
    assert (len(frequencies) == len(tps_values))

    fig = plt.figure(figsize=(10, 7.5))
    ax = plt.axes()

    if tps_values_2 is not None:
        # Include both plots
        assert (len(tps_values) == len(tps_values_2))
        ax.plot(frequencies, tps_values, '-o', label=label1, markersize=7.5, markerfacecolor='green',
                 linewidth=1.0)
        ax.plot(frequencies, tps_values_2, '-v', label=label2, markersize=7.5, markerfacecolor='red',
                 linewidth=1.0)
        ax.legend(fontsize=15)
    else:
        # Just include one plot
        ax.plot(frequencies, tps_values, '.', markersize=7.5)
        ax.plot(frequencies, tps_values, 'r', linewidth=1.0)

    if log_scale:
        ax.set_xscale('log')

    ax.set_title(title, fontsize=25, pad=10)
    ax.set_xlabel(x_label, fontsize=20, labelpad=10)
    ax.set_ylabel(y_label, fontsize=20, labelpad=10)
    ax.tick_params(labelsize=15, pad=5)
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)

    if show:
        plt.show()

    return fig


def compute_rms(data_values, mask=None):
    """
    Take the root mean-square (RMS) of a time series of 2-D arrays.

    Args:
        data_values (ndarray): numpy 2-D or 3-D array of shape (M, N) or (number of time-steps, M, N) containing the
            data.

        mask (ndarray, optional): [Default=None] numpy boolean 2-D of shape (M, N) specifying which pixels to use in the
            RMS calculation.

            - If set to None, all pixels are used.

    Returns:
        **rms** (*float*) -- the RMS of the data values.
    """
    assert (data_values.ndim in [2, 3])
    if mask is not None:
        assert (mask.shape == data_values.shape[1:])
        assert (mask.dtype == bool)
    else:
        mask = np.ones(data_values.shape[1:], dtype=bool)

    data_flat = aomodel.utils.img_to_vec(data_values, mask)
    rms = np.sqrt(np.average(data_flat ** 2))
    return rms


def compute_nrmse(ground_truth_data, estimated_data):
    """
    Compute the normalized root mean-square error (NRMSE) of estimated data with respect to ground-truth data. We use
    range normalization of the RMSE. The range is difference between the 95th and 5th percentiles of the ground-truth
    data.

    Args:
        ground_truth_data (ndarray): numpy array containing the ground-truth values.

        estimated_data (ndarray): numpy array containing the estimated values.

    Returns:
        **nrmse** (*float*) -- the NRMSE between the estimated and ground-truth data values.
    """
    data_error = ground_truth_data - estimated_data
    rmse = np.sqrt(np.average(data_error ** 2))
    percentile_95 = np.percentile(ground_truth_data, 95)
    percentile_5 = np.percentile(ground_truth_data, 5)
    value_range = percentile_95 - percentile_5
    nrmse = rmse / value_range
    return nrmse
