import numpy as np

# Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2026-1309.

def compressed_indexing_array(prediction_window_mask):
    """
    Compress the prediction window indices array (used in ``LongRangeAR``) to its smallest form. If different rows of
    the indices array have different numbers of elements, the function fills in the remaining (un-used) indices of
    smaller rows (i.e., rows with fewer prediction weights) with -1. This ensures that the compressed indexing array is
    compatible with the calculations in ``least_squares_solution`` and data generation algorithm used by
    ``LongRangeAR``.

    Args:
        prediction_window_mask (ndarray): numpy boolean 2-D array of shape
            (vector dimensionality, vector dimensionality * total lags) indicating which previous vector components to
            use for predicting the next vector. The i-th row of prediction_window_indices indicates which previous
            vector components to use for predicting the i-th component of the next vector.

    Returns:
        **prediction_window_indices** (*ndarray*) -- numpy 2-D integer-valued array of shape
        (number of predicted vector components, maximum prediction window size) containing the indices for which we
        calculate prediction weights.
    """
    assert (prediction_window_mask.ndim == 2)
    assert (prediction_window_mask.dtype == bool)

    predicted_comps = np.sort(np.unique(prediction_window_mask.nonzero()[0]))

    num_weights = np.sum(prediction_window_mask, axis=1)
    if num_weights.size == 0:
        max_num_weights = 0
    else:
        max_num_weights = np.max(num_weights)

    num_predicted_comps = len(predicted_comps)

    # In the case that there are no assigned prediction weights, returns an array compatible to the functionality of
    # LongRangeAR:
    if max_num_weights == 0:
        return np.full((num_predicted_comps, 0), -1, dtype=int)

    # Indexing map for valid entries of the prediction weights 2-D array:
    valid_row_indices_of_prediction_weights_array = np.repeat(np.arange(num_predicted_comps),
                                                              num_weights[num_weights.nonzero()])
    valid_col_indices_of_prediction_weights_array = np.concatenate([np.arange(val) for val in
                                                                    num_weights])
    valid_2d_indices_of_prediction_weights_array = (valid_row_indices_of_prediction_weights_array,
                                                    valid_col_indices_of_prediction_weights_array)

    # Index mapping from compressed prediction window array to full prediction window array:
    valid_prediction_window_indices = prediction_window_mask.nonzero()
    compressed_prediction_window_indexing_map = (np.full((num_predicted_comps, max_num_weights), -1,
                                                         dtype=np.int_),
                                                 np.full((num_predicted_comps, max_num_weights), -1,
                                                         dtype=np.int_))
    compressed_prediction_window_indexing_map[0][valid_2d_indices_of_prediction_weights_array] = \
        valid_prediction_window_indices[0]
    compressed_prediction_window_indexing_map[1][valid_2d_indices_of_prediction_weights_array] = \
        valid_prediction_window_indices[1]

    prediction_window_indices = compressed_prediction_window_indexing_map[1]

    return prediction_window_indices


def validate_prediction_window_indices(prediction_window_indices, vector_dimensionality, num_time_lags,
                                       num_low_pass_filters):
    """
    Checks if an input array ``prediction_window_indices`` is compatible with calculations in ``least_squares_solution``
    and data generation algorithm used by ``LongRangeAR``. Based on the creation of the ``prediction_window_indices``
    array in the function ``compressed_indexing_arrays``.

    Args:
        prediction_window_indices (ndarray): numpy 2-D integer-valued array of shape
            (number of predicted vector components, maximum prediction window size) containing the indices for which to
            calculate prediction weights. In the case that one vector component has a smaller prediction window size
            than others, the remaining spaces in the row associated to the component should be filled in with -1 (to the
            right of the true prediction window indices).

        vector_dimensionality (int): length of data vectors.

        num_time_lags (int): number of time-lags used in the prediction window.

        num_low_pass_filters (int): number of low-pass filters used in the prediction window.

    Raises:
        ValueError: if the input array ``prediction_window_indices`` is incompatible with the package.
    """
    total_lags = num_time_lags + num_low_pass_filters
    max_col = total_lags * vector_dimensionality - 1

    # 1) Domain: ensures that values are either -1 (invalid index) or [0, max_col] (valid indices):
    invalid = ~((prediction_window_indices == -1) |
                ((0 <= prediction_window_indices) & (prediction_window_indices <= max_col)))
    if np.any(invalid):
        bad = np.unique(prediction_window_indices[invalid])
        raise ValueError(f"prediction_window_indices contains out-of-range values: {bad}.")

    # 2) Row non-emptiness: ensures that there is at least one valid column index per row:
    if np.any(np.all((prediction_window_indices == -1), axis=1)):
        bad_rows = np.where(np.all((prediction_window_indices == -1), axis=1))[0]
        raise ValueError(f"Rows {bad_rows.tolist()} of prediction_window_indices have no valid regressors (all -1).")

    # 3) Contiguous padding per row: if there is a -1 entry in a given row, ensure all entries to its right are also -1:
    num_columns = prediction_window_indices.shape[1]
    if num_columns > 0:
        first_pad = (
            np.where((prediction_window_indices == -1), np.arange(num_columns)[None, :], num_columns).min(axis=1))
        mask_right = (np.arange(num_columns)[None, :] >= first_pad[:, None])
        row_ok = np.all(((~mask_right) | (prediction_window_indices == -1)), axis=1)
        if not np.all(row_ok):
            bad_rows = np.where(~row_ok)[0]
            raise ValueError( f"Rows {bad_rows.tolist()} of prediction_window_indices have non-contiguous padding; "
                              "valid entries must be left-packed with -1s to the right.")

    # 4) Low-pass filter mask uniform across rows (ignoring -1): if low-pass filters are used, ensure that the column
    # positions are consistent across all rows
    if num_low_pass_filters > 0:
        lag_block = prediction_window_indices // vector_dimensionality
        lpf_mask = (lag_block < num_low_pass_filters) & (prediction_window_indices != -1)
        row0 = lpf_mask[0]
        same = np.array([np.array_equal(row0, lpf_mask[r]) for r in range(prediction_window_indices.shape[0])])
        if not np.all(same):
            bad_rows = np.where(~same)[0]
            raise ValueError("Low-pass filter column positions differ across rows of prediction_window_indices; "
                             f"rows {bad_rows.tolist()} do not match row 0. "
                             "Generation requires uniform Low-pass filter column positions.")


def prediction_window_indexing_arrays(vector_dimensionality, time_lags, prediction_window_indices,
                                      num_low_pass_filters=0):
    """
    Construct the indexing arrays used for model fitting and synthetic data generation in ``LongRangeAR``.

    Args:
        vector_dimensionality (int): length of data vectors.

        time_lags (Union[int, list, ndarray]): either an integer, list, or numpy 1-D array indicating the time lags to
            use for the model.

            - If ``time_lags`` is an integer, the function uses the previous time-steps up to this integer value.

            - If ``time_lags`` is a list or numpy 1-D array, the model uses these time lags.

        prediction_window_indices (ndarray): numpy 2-D integer-valued array of shape
            (number of predicted vector components, maximum prediction window size) containing the indices for which we
            calculate prediction weights. In the case that one vector component has a smaller prediction window size
            than others, the remaining spaces in the row associated to the component should be filled in with -1 (to the
            right of the true prediction window indices).

        num_low_pass_filters (int, optional): [Default=0] number of low-pass filters used in the prediction window.

    Returns:
        - **time_step_offsets_for_prediction_window** (*ndarray*) -- numpy 2-D array of variable shape (depending on
          the input ``prediction_window_indices``) containing the time-index offsets associated to each prediction
          weight.
        - **vector_component_indices_for_prediction_window** (*ndarray*) -- numpy 2-D array of variable shape
          (also depending on the input ``prediction_window_indices``) containing the vector component indices associated
          to each prediction weight.
        - **lpf_column_index_map** (*ndarray*) -- numpy 1-D array mapping the low-pass filter columns of the array
          ``time_step_offsets_for_prediction_window`` to the associated low-pass filter index. If no low-pass filters
          are used, the array is empty.
    """
    time_lags = np.array(time_lags)
    assert (time_lags.ndim <= 1)
    if time_lags.ndim == 0:
        assert (time_lags > 0)
        time_lags = np.arange(1, time_lags + 1)
    else:
        assert (time_lags > 0).all()
    assert (num_low_pass_filters >= 0)

    # Ensure a valid prediction window:
    prediction_window_indices = np.array(prediction_window_indices, dtype=int)
    validate_prediction_window_indices(prediction_window_indices, vector_dimensionality, len(time_lags),
                                       num_low_pass_filters)

    # Full array of vector component indices
    vector_component_indices_for_prediction_window = prediction_window_indices % vector_dimensionality

    # If low-pass filters are used, then creates room in the time-step offset indexing matrices for low-pass filter
    # columns.
    lpf_column_index_map = np.array([], dtype=int)
    if num_low_pass_filters > 0:
        # Low-pass filter columns will have negative values:
        time_lag_indices = (prediction_window_indices // vector_dimensionality) - num_low_pass_filters
        lpf_mask = time_lag_indices < 0

        # Extracts low-pass filter column indices:
        num_lpf_columns = int(np.sum(lpf_mask[0]))

        # The 1-D array of low-pass filter IDs for each column:
        lpf_id_per_column = time_lag_indices[0, :num_lpf_columns]

        # Index map: Maps the low-pass filter column to the index of the associated low-pass filter:
        lpf_column_index_map = num_low_pass_filters + lpf_id_per_column

        # Array of time-step offsets:
        time_step_offsets_for_prediction_window = np.empty_like(prediction_window_indices)

        # Place positive place-holders in the low-pass filter columns:
        time_step_offsets_for_prediction_window[lpf_mask] = -time_lag_indices[lpf_mask]

        # Use the correct time-step offsets (i.e., negative of time-lag value) for the AR columns:
        standard_time_lag_mask = (~lpf_mask & (time_lag_indices >= 0))
        time_step_offsets_for_prediction_window[standard_time_lag_mask] = \
            -time_lags[time_lag_indices[standard_time_lag_mask]]
    else:
        time_step_offsets_for_prediction_window = -time_lags[(prediction_window_indices // vector_dimensionality)]

    # For invalid indices, put in placeholder indices (which won't actually be used):
    invalid_indices = (prediction_window_indices == -1)
    if np.any(invalid_indices):
        vector_component_indices_for_prediction_window[invalid_indices] = vector_dimensionality - 1
        time_step_offsets_for_prediction_window[invalid_indices] = -time_lags[-1]

    return time_step_offsets_for_prediction_window, vector_component_indices_for_prediction_window, lpf_column_index_map
