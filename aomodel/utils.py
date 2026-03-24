import numpy as np


def vec_to_img(data_vec, mask):
    """
    Reshapes row vector(s) into 2-D array(s) ("images") in raster order using a mask. The values of the returned data
    outside of the mask are set to float("nan"). The function can convert either a vector to a single image or a
    2-D array (containing a sequence of vectors) to a sequence of images (in a 3-D array).

    Args:
        data_vec (ndarray): numpy 1-D or 2-D array with the input vector(s).

            - If ``data_vec`` is 1-D, a single vector is converted to a single image.

            - If ``data_vec`` is 2-D, it has shape (number of images, image dimensionality) and contains multiple vectors.

        mask (ndarray): numpy 2-D boolean array of shape (image height, image width) indicating which 2-D indices of
            each time-step correspond to valid pixel values. The number of values in the mask that are set to "True"
            must equal the dimensionality of the input data vector(s).

    Returns:
        **output_image** (*ndarray*) -- numpy 2-D or 3-D array containing the image(s). If ``data_vec`` is 1-D, only a
        single 2-D array of shape (image height, image width) is returned. If ``data_vec`` is 2-D, a 3-D array of shape
        (number of images, image height, image width) is returned.
    """
    assert (mask.sum() == data_vec.shape[-1])
    assert (1 <= len(data_vec.shape) <= 2)

    # Determines if we are converting to a single image or a sequence of images:
    if len(data_vec.shape) == 1:  # converting to a single image
        output_shape = [1, mask.shape[0], mask.shape[1]]
    else:  # converting to a sequence of images
        output_shape = [data_vec.shape[0], mask.shape[0], mask.shape[1]]

    # Output image array:
    output_image = np.full(output_shape, float("nan"))

    # Sets the valid indices to the corresponding data value:
    output_image[:, mask] = data_vec

    return output_image.squeeze()


def img_to_vec(image_data, mask):
    """
    Reshapes 2-D array(s) ("image(s)") into row vector(s) in raster order using a mask. The function can reshape either
    a single image into a single vector or a 3-D array (containing a sequence of images) into a sequence of rows vectors
    (i.e., a 2-D array).

    Args:
        image_data (ndarray): numpy 2-D or 3-D array containing the images.

            - If ``image_data`` is 2-D, a single image is reshaped into a single vector. In this case, the input must
              have shape (image height, image width).

            - If ``image_data`` is 3-D, a sequence of images is reshaped into a sequence of row vectors. In this case,
              the input must have shape (number of images, image height, image width).

        mask (ndarray): numpy 2-D boolean array of shape (image height, image width) indicating which 2-D indices of
            each time-step correspond to valid pixel values. Only indices set to "True" will be included in the output
            vectors.

    Returns:
        **output_vector** (*ndarray*) -- numpy 1-D or 2-D array with the vector(s) (i.e., flattened image(s)). If
        ``image_data`` is 2-D, only a single 1-D array is returned. If ``image_data`` is 3-D, a 2-D array of shape
        (number of images, image dimensionality) is returned.
    """
    assert (2 <= len(image_data.shape) <= 3)
    # Determines if we are converting a single image or a sequence of images:
    if len(image_data.shape) == 2:  # converting a single image
        assert (image_data.shape == mask.shape)
        output_vector = image_data[mask]
    else:
        assert (image_data.shape[1:] == mask.shape)
        # Converting a sequence of images
        output_vector = image_data[:, mask]

    return output_vector


def parabolic_interpolation_max(input_vals, function_vals):
    """
    Uses parabolic interpolation to estimate the input to a function which gives the largest output.

    Args:
        input_vals (ndarray): numpy 1-D array of input values.

        function_vals (ndarray): numpy 1-D array of function values at the corresponding inputs.

    Returns:
        **max_input** (*float*) -- the estimate of the input resulting in the peak function value.
    """
    assert (input_vals.ndim == 1)
    assert (input_vals.shape == function_vals.shape)

    maximal_index = np.squeeze(np.argmax(function_vals))

    if (maximal_index == 0) or (maximal_index == len(input_vals) - 1):
        max_input = input_vals[maximal_index]
    else:
        # Fits a parabola to the three values centered at the maximum:
        neighboring_inputs = input_vals[maximal_index - 1:maximal_index + 2]
        neighboring_function_vals = function_vals[maximal_index - 1:maximal_index + 2]
        polynomial_coefficients = np.polyfit(neighboring_inputs, neighboring_function_vals, 2)
        a, b, c = polynomial_coefficients
        max_input = -b / (2 * a)

    return max_input
