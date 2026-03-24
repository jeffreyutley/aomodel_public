import numpy as np
import time
from datetime import timedelta
import aomodel.pca as pca
import os
import warnings
import aomodel._indexing as indexing
from aomodel.prediction_window_structure import *

# Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2026-1309.

class LongRangeAR:
    """
    Implements a **Long-Range Auto-Regressive** model for generating synthetic time series of vectors with the same
    spatial and temporal statistics as an input data set. For a thorough description of this model, see :ref:`Theory`.
    This implementation of Long-Range AR is partially adapted from a Vector AR model,  which is described in
    :cite:`Lutkepohl`.

    Args:
        vector_dimensionality (int, optional): [Default=None] length of the data vectors.

            - If set to None, then a ``load_file`` must be provided. In this case, the class loads the vector dimension
              from the file.

        time_lags (Union[int, list, ndarray], optional): [Default=1] either an integer, list, or numpy 1-D array
            indicating the time lags to use for the model.

            - If ``time_lags`` is an integer, the function uses the previous time-steps up to this integer value.

            - If ``time_lags`` is a list or numpy 1-D array, the model uses these time lags.

        num_low_pass_filters (int, optional): [Default=0] the number of low-pass filters to use in the linear time
            predictive model.

        load_file (str, optional): [Default=None] directory to a file from which the model's instance variables can
            be loaded.

        prediction_window_structure (PredictionWindowStructure, optional): [Default=None] an instance of a sub-class
            of ``PredictionWindowStructure`` implementing get_mask(). Passed as input to
            ``self.define_prediction_window``.

            - If set to None, the method builds a prediction window structure from `**kwargs`.

        print_statements (str, optional): [Default=True] whether to print out ``LongRangeAR`` statements.

        **kwargs (dict, optional): Keyword arguments for legacy support.

            - **prediction_window_distance** (*int*) -- an index distance used to define the prediction window. Passed
              as input to ``self.define_prediction_window``.

            - **prediction_subspace_dimension** (*int*) -- number of (top) vector components to use for linear time
              prediction. If included, the function uses this subspace as the prediction window. A prediction
              subspace and a prediction window distance cannot both be used. Passed as input to
              ``self.define_prediction_window``.

            - **low_pass_filter_params** (*ndarray*) -- numpy 1-D array of length ``num_low_pass_filters`` containing
              the parameters of each low-pass filter.
    """

    def __init__(self, vector_dimensionality=None, time_lags=1, num_low_pass_filters=0, load_file=None,
                 prediction_window_structure=None, print_statements=True, **kwargs):
        if vector_dimensionality is None:
            assert (load_file is not None)
        else:
            assert (vector_dimensionality > 0)

        if print_statements:
            print("\nInitializing Long-Range AR model...")

        # Core variables
        self.time_lags = self._validate_time_lags(time_lags)
        self.vector_dimensionality = vector_dimensionality
        self.low_pass_filter_params = self._validate_low_pass_filters(num_low_pass_filters_arg=num_low_pass_filters,
                                                                      options_dictionary=kwargs)
        kwargs.pop('low_pass_filter_params', None)

        # Remaining variables initialized to None:
        self.prediction_window_distance = None
        self.prediction_subspace_dimension = None
        self.prediction_weights = None
        self._base_prediction_window_mask = None
        self.prediction_window_indices = None
        self.residuals_mean = None
        self.noise_modulation = None
        self._is_fitted = False

        # Uses the load_file (if input by the user) to load necessary instance variables:
        if load_file is not None:
            self.load(load_file=load_file, print_statements=print_statements)
        else:
            # Uses the method 'define_prediction_window' to initialize many of the remaining variables:
            self.create_model_structure(prediction_window_structure=prediction_window_structure, **kwargs)
            if print_statements:
                print("Long-Range AR model initialized. Use LongRangeAR.fit() to fit this model to data or "
                      "LongRangeAR.load() to load a pre-trained model.\n")

    def fit(self, training_data, cutoff_frequency=None, tps_block_size=None, print_statements=True):
        """
        Calculates the model prediction weights and noise modulation matrix using least squares regression from the
        input data values and Principal Component Analysis (respectively).

        Args:
            training_data (ndarray): numpy 2-D array of shape (data vector dimensionality, number of time samples)
                containing the data to fit.

            cutoff_frequency (float, optional): [Default=None] cutoff frequency (in units of cycles/sample) to use for
                estimation of the low-pass filter parameters.

                - If set to None and low-pass filters are used, this method estimates the cutoff frequency as the peak
                  frequency of the temporal power spectrum (TPS) of the training data.

            tps_block_size (int, optional): [Default=None] time-block size for calculating the TPS of the data.

                - If set to None and the low-pass filter parameters need to be estimated, the function finds the block
                  size so that the TPS estimates are averaged over 100 blocks. This ensures that the signal-to-noise
                  ratio of each 1-D TPS estimate is at least 10.

            print_statements (str, optional): [Default=True] whether to print out ``LongRangeAR`` statements.
        """
        self.__validate_training_data(training_data=training_data)

        # Warns about overwriting previously-fit models:
        if self._is_fitted:
            warnings.warn("This model instance has already been fitted. Calling fit() will overwrite existing "
                          "parameters.", UserWarning)

        from aomodel.estimation import estimate_long_range_ar_parameters

        # Updates the data vector dimensionality for this model:
        num_time_steps = training_data.shape[1]

        if print_statements:
            start_time = time.time()
            print("\nLong-Range AR Model Fitting\n"
                  "===========================\n"
                  "Number of time-steps in input data:  %d" % num_time_steps)

        # Use the designated Long-Range AR parameter estimation function:
        parameters = estimate_long_range_ar_parameters(
            training_data=training_data,
            time_lags=self.time_lags,
            prediction_window_indices=self.prediction_window_indices,
            predicted_components=self.predicted_components,
            num_low_pass_filters=self.num_low_pass_filters,
            low_pass_filter_params=self.low_pass_filter_params,
            cutoff_frequency=cutoff_frequency,
            tps_block_size=tps_block_size
        )

        # Save estimated parameters:
        prediction_weights = parameters["prediction_weights"]
        self.low_pass_filter_params = parameters['low_pass_filter_params']
        self.residuals_mean = parameters['residuals_mean']
        self.noise_modulation = parameters['noise_modulation']

        # Saves the least-squares solution values as prediction weights at the corresponding indices of the 2-D
        # prediction weights array (i.e., at indices that should contain non-zero values):
        valid_indices_of_prediction_weights_array = (self.prediction_window_indices != -1)
        self.prediction_weights[valid_indices_of_prediction_weights_array] = prediction_weights.flatten()

        self._is_fitted = True

        if print_statements:
            elapsed_time = str(timedelta(seconds=time.time() - start_time))
            print(f"Long-Range AR model fitting completed in {elapsed_time} (hr:min:sec)\n")

            # Total number of parameters estimated from data:
            print(f"Total number of parameters: ", self.num_parameters)

    def run(self, num_time_steps, initial_vectors=None, print_statements=True):
        """
        Uses a trained **Long-Range AR** model (as set by either ``self.fit`` or ``self.load``) to generate synthetic
        data with the same temporal statistics as the training data.

        Args:
            num_time_steps (int): the number of time-steps of synthetic data vectors to generate.
            initial_vectors (ndarray, optional): [Default=None] numpy 2-D array of shape (data vector dimensionality,
                number of time lags) containing the initial data vectors used by the linear predictive model.

                - If set to None, vectors of zeros are used.

            print_statements (str, optional): [Default=True] whether to print out ``LongRangeAR`` statements.

        Returns:
            **synthetic_data** (*ndarray*) -- a 2-D numpy array of shape (data vector dimensionality,
            ``num_time_steps``) containing the model's output synthetic data.

        Raises:
            RuntimeError: if the attribute ``self._if_fitted`` is not True (indicating that the model has not been fit).
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Please call 'fit()' or 'load()' before running generation.")

        assert (num_time_steps > 0)

        if initial_vectors is not None:
            assert (initial_vectors.shape == (self.vector_dimensionality, max(self.time_lags)))
        else:
            initial_vectors = np.zeros((self.vector_dimensionality, max(self.time_lags)))

        if print_statements:
            start_time = time.time()
            print("\nLong-Range AR Model Generation\n"
                  "==============================\n"
                  f"Generating {num_time_steps} time-steps of synthetic data...\n")

        # Finds the indices of vector components for which there are no prediction weights:
        predicted_components = self.predicted_components
        remaining_comps = np.setdiff1d(np.arange(0, self.vector_dimensionality), predicted_components)

        # Create indexing arrays to use for data generation:
        (time_step_offsets_for_prediction_window,
         vector_component_indices_for_prediction_window,
         lpf_column_index_map) = self._indexing_arrays_for_data_generation()
        num_lpf_columns = len(lpf_column_index_map)

        # Initializes the output 'synthetic_data' array and fills in the initial vectors:
        synthetic_data = np.zeros((self.vector_dimensionality, num_time_steps + max(self.time_lags)))
        synthetic_data[:, :max(self.time_lags)] = initial_vectors

        # Uses PCA to generate synthetic prediction error (or residuals) for AR synthesis:
        input_noise = pca.generative_pca_algorithm(num_samples=num_time_steps,
                                                   covariance_modulation_matrix=self.noise_modulation,
                                                   mean_vector=self.residuals_mean)

        # If low-pass filters are used by this model instance, initializes arrays to store the low-pass filters:
        if num_lpf_columns > 0:
            # Place the low-pass filter parameters into arrays:
            low_pass_input_contribution_weights = self.low_pass_filter_params[lpf_column_index_map][None, :]
            low_pass_memory_retention_weights = 1.0 - low_pass_input_contribution_weights

            # Array of vector component indices used an input to each low-pass filter.
            lpf_vector_component_input_indices = vector_component_indices_for_prediction_window[:, :num_lpf_columns]

            # 2-D array that holds all low-pass filter states:
            low_pass_filter_state = np.zeros((len(predicted_components), num_lpf_columns))

        # Applies the Long-Range AR model recursively for the vector components that have been assigned (non-zero)
        # prediction weights:
        for j in range(max(self.time_lags), num_time_steps + max(self.time_lags)):
            # Pulls the previous values (to use for linear time prediction) into an array:
            if num_lpf_columns > 0:
                # Previous data values from the prediction window:
                previous_data_vector_values = (
                    synthetic_data[vector_component_indices_for_prediction_window[:, num_lpf_columns:],
                    j + time_step_offsets_for_prediction_window[:, num_lpf_columns:]])

                # Combines the previous data values and low-pass filters in the array:
                previous_values = np.hstack([low_pass_filter_state, previous_data_vector_values])
            else:
                previous_values = \
                    synthetic_data[vector_component_indices_for_prediction_window, j + time_step_offsets_for_prediction_window]

            # Computes the linear time prediction:
            synthetic_data[predicted_components, j] = \
                np.sum(previous_values * self.prediction_weights, axis=1) + \
                input_noise[(predicted_components, j - max(self.time_lags))]

            # Updates the low-pass filters:
            if num_lpf_columns > 0:
                new_data_values = synthetic_data[lpf_vector_component_input_indices, j]
                low_pass_filter_state = (low_pass_memory_retention_weights * low_pass_filter_state +
                                         low_pass_input_contribution_weights * new_data_values)

        # Fill in the noise for the remaining (non-predicted) components:
        synthetic_data[remaining_comps, max(self.time_lags):] = input_noise[remaining_comps]

        if print_statements:
            elapsed_time = str(timedelta(seconds=time.time() - start_time))
            print("Long-Range AR Model Generation Completed in {} (hr:min:sec)\n".format(elapsed_time))
        return synthetic_data[:, max(self.time_lags):]

    def save(self, save_file):
        """
        Saves all necessary **Long-Range AR** model information to re-construct the trained ``LongRangeAR`` instance.

        Args:
            save_file (str): directory of the file to which the data will be saved.
        """
        assert (type(save_file) == str)
        print(f"Saving Long-Range AR model to {os.path.abspath(save_file)}...")
        valid_indices_of_prediction_weights_array = (self.prediction_window_indices != -1)
        prediction_weights = self.prediction_weights[valid_indices_of_prediction_weights_array]

        save_arrays = {'noise_modulation': self.noise_modulation, 'residuals_mean': self.residuals_mean,
                       'prediction_window_mask': self._base_prediction_window_mask, 'time_lags': self.time_lags,
                       'prediction_weights': prediction_weights}

        if self.prediction_window_distance is not None:
            save_arrays['prediction_window_distance'] = self.prediction_window_distance

        if self.prediction_subspace_dimension is not None:
            save_arrays['prediction_subspace_dimension'] = self.prediction_subspace_dimension

        if self.num_low_pass_filters > 0:
            save_arrays['low_pass_filter_params'] = self.low_pass_filter_params

        np.savez(save_file, **save_arrays)
        print(f"Long-Range AR model saved to {os.path.abspath(save_file)}.\n")

    def load(self, load_file, print_statements=True):
        """
        Loads the **Long-Range AR** model information as saved by the ``self.save`` method and re-constructs the model.

        Args:
            load_file (str): directory of the file from which the data will be loaded.
            print_statements (str, optional): [Default=True] whether to print out model loading statements.
        """
        assert (type(load_file) == str)

        if print_statements:
            print(f"Loading LongRangeAR model from {load_file}...")

        data = np.load(file=load_file, allow_pickle=True)

        time_lags = data['time_lags']
        prediction_window_mask = data['prediction_window_mask']
        prediction_weights = data['prediction_weights']

        prediction_window_distance = None
        if 'prediction_window_distance' in data:
            prediction_window_distance = data['prediction_window_distance']
            if hasattr(prediction_window_distance, 'item'):
                prediction_window_distance = prediction_window_distance.item()

        prediction_subspace_dimension = None
        if 'prediction_subspace_dimension' in data:
            prediction_subspace_dimension = data['prediction_subspace_dimension']
            if hasattr(prediction_subspace_dimension, 'item'):
                prediction_subspace_dimension = prediction_subspace_dimension.item()

        if 'low_pass_filter_params' in data:
            low_pass_filter_params = data['low_pass_filter_params']
        else:
            low_pass_filter_params = np.zeros(0)

        self.vector_dimensionality = prediction_window_mask.shape[0]
        self.noise_modulation = data['noise_modulation']
        self.residuals_mean = data['residuals_mean']

        if print_statements:
            print(f"Finished loading LongRangeAR model from {load_file}.\n")

        self.create_model_structure(prediction_window_mask=prediction_window_mask, time_lags=time_lags,
                                    low_pass_filter_params=low_pass_filter_params)

        if prediction_window_distance is not None:
            self.prediction_window_distance = prediction_window_distance
        if prediction_subspace_dimension is not None:
            self.prediction_subspace_dimension = prediction_subspace_dimension

        valid_indices_of_prediction_weights_array = (self.prediction_window_indices != -1)
        self.prediction_weights[valid_indices_of_prediction_weights_array] = prediction_weights

        self._is_fitted = True

        # Total number of parameters estimated from data:
        if print_statements:
            print(f"Total number of parameters: ", self.num_parameters, '\n')

    def create_model_structure(self, prediction_window_structure=None, **kwargs):
        """
        Creates the linear predictive model structure by specifying:

            1. The "prediction window" indicating which components of previous vectors to use for linear prediction
               (i.e., for which the model estimates prediction weights).

            2. The time-lags (i.e., backwards time-index shifts) to use for next-state prediction.

            3. The low-pass filter structure to use in next-state prediction. This includes both the number of low-pass
               filters and the parameters of each filter.

        Note:
            Calling this function re-defines the model structure. If there is already an existing structure including
            prediction weights and low-pass filter parameters, calling this function resets both of these objects. Only
            call this function if you intend to then call self.fit to estimate the prediction weights and low-pass
            filter parameters from data.

        Args:
            prediction_window_structure (PredictionWindowStructure, optional): [Default=None] an instance of a sub-class
                of ``PredictionWindowStructure`` implementing get_mask(). Used to define the prediction window mask.

                - If set to None, the method builds a prediction window structure from `**kwargs`.

            **kwargs (dict, optional): Keyword arguments for prediction window structure configuration:

                - **prediction_window_mask** (*ndarray*) -- numpy boolean 2-D array of shape
                  (data vector dimensionality, data vector dimensionality) indicating which prediction weights to solve
                  for in the model fitting. Remaining weights are automatically set to zero. This mask determines the
                  prediction mask for each time-lag and low-pass filter.

                - **prediction_window_distance** (*int*) -- an index distance used to define the prediction window.

                - **prediction_subspace_dimension** (*int*) -- number of (top) vector components to use for linear time
                  prediction. If included, the function uses this subspace as the prediction window. A prediction
                  subspace and a prediction window distance cannot both be used.

                - **predicted_components** (*ndarray*) -- numpy 1-D integer array containing the indices of the data
                  vector components for which the model will compute prediction weights.

                - **time_lags** (*Union[int, list, ndarray, None]*) -- either an integer, list, or numpy 1-D array
                  indicating the time lags to use for the model. If ``time_lags`` is an integer, the function uses the
                  previous time-steps up to this integer value. If ``time_lags`` is a list or numpy 1-D array, the model
                  uses these time lags.

                - **num_low_pass_filters** (*int*) -- the number of low-pass filters to use in the linear time
                  predictive model.

        Raises:
            ValueError: if both ``prediction_window_distance`` and ``prediction_subspace_dimension`` are used.
        """
        predicted_components = kwargs.get('predicted_components')
        predicted_components = self._validate_predicted_components(predicted_components)

        if self._base_prediction_window_mask is not None:
            print("\n|>   Updating Long-Range AR Model Parameters...\n|>")
        else:
            print("\n|>   Initializing Long-Range AR Model Parameters...\n|>")

        # If no prediction window structure is provided, initialize one using inputs in **kwargs:
        if prediction_window_structure is None:
            if ((kwargs.get('prediction_window_distance') is not None) and
                    (kwargs.get('prediction_subspace_dimension') is not None)):
                raise ValueError(
                    "Cannot specify both 'prediction_window_distance' and 'prediction_subspace_dimension'.")

            if kwargs.get('prediction_window_mask') is not None:
                prediction_window_structure = ExplicitWindowStructure(kwargs['prediction_window_mask'])
            elif kwargs.get('prediction_subspace_dimension') is not None:
                prediction_window_structure = SubspaceWindowStructure(kwargs['prediction_subspace_dimension'])
            elif kwargs.get('prediction_window_distance') is not None:
                prediction_window_structure = DistanceWindowStructure(kwargs['prediction_window_distance'],
                                                                      predicted_components)
            else:
                prediction_window_structure = FullWindowStructure(predicted_components)

        self.prediction_window_distance = None
        self.prediction_subspace_dimension = None

        if isinstance(prediction_window_structure, DistanceWindowStructure):
            self.prediction_window_distance = prediction_window_structure.distance
        elif isinstance(prediction_window_structure, SubspaceWindowStructure):
            self.prediction_subspace_dimension = prediction_window_structure.subspace_dimension

        # Ensure the time-lags input is valid:
        if ('time_lags' in kwargs) and (kwargs['time_lags'] is not None):
            self.time_lags = self._validate_time_lags(kwargs['time_lags'])

        # Initialize the low pass filter parameters if necessary:
        num_low_pass_filters = kwargs.get('num_low_pass_filters')
        self.low_pass_filter_params = self._validate_low_pass_filters(num_low_pass_filters_arg=num_low_pass_filters,
                                                                      options_dictionary=kwargs)
        num_low_pass_filters = self.num_low_pass_filters

        # Base mask contains the prediction window mask associated to a single time-lag:
        self._base_prediction_window_mask = prediction_window_structure.get_mask(self.vector_dimensionality)

        # Compresses the prediction window mask and extracts the indices corresponding to this compressed array:
        self.prediction_window_indices = indexing.compressed_indexing_array(self.prediction_window_mask)

        # Initializes the prediction weights array:
        self.prediction_weights = np.zeros(self.prediction_window_indices.shape)

        # Print out model parameters
        print("|>   Long-Range AR Model Parameters\n"
              "|>   ==========================================\n"
              "|>   " + "{:<31}".format("Vector Dimensionality") + "|%8s" % self.vector_dimensionality + " |\n"
              "|>   " + "{:<31}".format("Number of Time Lags") + "|%8s" % len(self.time_lags) + " |")

        if num_low_pass_filters > 0:
            print("|>   " + "{:<31}".format("Number of Low-Pass Filters") + "|%8s" % num_low_pass_filters + " |")

        if self.prediction_window_distance is not None:
            print("|>   " + "{:<31}".format("Prediction Window Distance") + "|%8s" %
                  self.prediction_window_distance + " |")

        if self.prediction_subspace_dimension is not None:
            print("|>   " + "{:<31}".format("Prediction Subspace Dimension") + "|%8s" %
                  self.prediction_subspace_dimension + " |")

        print("|>   " + "{:<31}".format("Number of Predicted Components") + "|%8s" % len(self.predicted_components) + " |\n"
              "|>   " + "{:<31}".format("Number of Prediction Weights") + "|%8s" % self.num_prediction_weights + " |\n"
              "|>   " + "==========================================\n")

    def _indexing_arrays_for_data_generation(self):
        """
        Uses the instance variables to create the indexing arrays used to generate synthetic data.

        Returns:
            - **time_step_offsets_for_prediction_window** (*ndarray*) -- numpy 2-D array of variable shape (depending on
              the attribute ``self.prediction_window_indices``) containing the time-index offsets associated to each
              prediction weight.
            - **vector_component_indices_for_prediction_window** (*ndarray*) -- numpy 2-D array of variable shape
              (also depending on the attribute ``self.prediction_window_indices``) containing the vector component
              indices associated to each prediction weight.
            - **lpf_column_index_map** (*ndarray*) -- numpy 1-D array mapping the low-pass filter columns of the array
              ``time_step_offsets_for_prediction_window`` to the associated low-pass filter index. If no low-pass
              filters are used, the array is empty.
        """
        return indexing.prediction_window_indexing_arrays(
            vector_dimensionality=self.vector_dimensionality,
            time_lags=self.time_lags,
            prediction_window_indices=self.prediction_window_indices,
            num_low_pass_filters=self.num_low_pass_filters)

    def _validate_time_lags(self, time_lags):
        """
        Validates and standardizes the ``time_lags`` input.

        Args:
            time_lags (Union[int, list, ndarray]): either an integer, list, or numpy 1-D array indicating the time lags
                to use for the model.

                - If ``time_lags`` is an integer, the function uses the previous time-steps up to this integer value.

                - If ``time_lags`` is a list or numpy 1-D array, the model uses these time lags.

                - If set to None, the function uses the instance variable ``self.time_lags`` for the value of time_lags.

        Returns:
            **time_lags** (*Union[ndarray, None]*) -- numpy 1-D array containing the explicit time-lags or None.

        Raises:
            ValueError: If dimensions, types, or values are invalid.
        """
        if time_lags is None:
            return self.time_lags

        time_lags = np.array(time_lags)

        if time_lags.ndim > 1:
            raise ValueError(f"time_lags must be 1-dimensional, got shape {time_lags.shape}")

        if not np.issubdtype(time_lags.dtype, np.integer):
            raise ValueError(f"time_lags must be an integer, got {time_lags.dtype}")

        if time_lags.ndim == 0:
            if time_lags <= 0:
                raise ValueError("When time_lags is an integer, it must be positive.")
            return np.arange(1, time_lags + 1)

        if not np.all(time_lags > 0):
            raise ValueError("All entries in time_lags must be strictly positive.")

        return time_lags

    def _validate_low_pass_filters(self, num_low_pass_filters_arg, options_dictionary):
        """
        Validates and standardizes the low-pass filters input.

        Args:
            num_low_pass_filters_arg (Union[int, None]): the number of low-pass filters to use in the linear time
                predictive model.

            options_dictionary (dict): dictionary of optional arguments passed as input to
                ``self.define_prediction_window``. This function considers the variable ``low_pass_filter_params``
                inside the dictionary:

                - **low_pass_filter_params** (*ndarray*): numpy 1-D array of length ``num_low_pass_filters`` containing
                  the parameters of each low-pass filter.

        Returns:
            **low_pass_filter_params** (*ndarray*) -- numpy 1-D array containing the low-pass filter parameters.

        Raises:
            ValueError: If dimensions or values are invalid.
        """
        # Case 1: Low pass filter parameters specified:
        if options_dictionary.get('low_pass_filter_params') is not None:
            low_pass_filter_params = np.array(options_dictionary['low_pass_filter_params'])
            if low_pass_filter_params.ndim == 0:
                low_pass_filter_params = np.array([low_pass_filter_params])
            if not ((low_pass_filter_params > 0).all() and (low_pass_filter_params < 1).all()):
                raise ValueError("low_pass_filter_params must be strictly between 0 and 1.")
            return low_pass_filter_params

        # Case 2: Parameters not specified, but number of low-pass filters is provided as an explicit argument.
        # Initializes low-pass filter parameters to zeros:
        if num_low_pass_filters_arg is not None:
            if num_low_pass_filters_arg < 0:
                raise ValueError("num_low_pass_filters must be non-negative.")
            return np.zeros(num_low_pass_filters_arg)

        # Case 3: Neither the parameters nor number of low-pass filters are specified.
        # Return the current low-pass filter parameters:
        return self.low_pass_filter_params

    def _validate_predicted_components(self, predicted_components):
        """
        Validates the ``predicted_components`` input.

        Args:
            predicted_components (ndarray or None): numpy 1-D integer array containing the indices of the data
                vector components for which the model will compute prediction weights.

        Returns:
            **predicted_components** (*Union[ndarray, None]*) -- numpy 1-D integer array containing the predicted
            components or None.

        Raises:
            ValueError: If dimensions, types, or values are invalid.
        """
        if predicted_components is None:
            return None

        predicted_components = np.array(predicted_components)

        if predicted_components.ndim != 1:
            raise ValueError(f"predicted_components must be 1-dimensional, got shape {predicted_components.shape}.")

        if not np.issubdtype(predicted_components.dtype, np.integer):
            raise ValueError(f"predicted_components must contain integers, got {predicted_components.dtype}.")

        if not ((0 <= predicted_components).all() and (predicted_components < self.vector_dimensionality).all()):
            raise ValueError(f"predicted_components must be within [0, {self.vector_dimensionality - 1}].")

        return predicted_components

    def __validate_training_data(self, training_data):
        """
        Ensures that the training data shape is compatible with the ``LongRangeAR`` class and the vector dimensionality
        of the model instance.

        Args:
            training_data (ndarray): numpy 2-D array of shape (data vector dimensionality, number of time samples)
                containing the data to fit.

        Raises:
            ValueError: if the training data is not compatible with the class or with ``self.vector_dimensionality``.
        """
        if training_data.ndim != 2:
            raise ValueError(f"Training data array must be 2-dimensional, got shape {training_data.shape}.")
        if training_data.shape[0] != self.vector_dimensionality:
            raise ValueError(f"Training data vectors must have the same dimension as self.vector_dimensionality,"
                             f"got vectors of length {training_data.shape[0]}.")
        if training_data.shape[1] == 0:
            raise ValueError(f"Training data vector must have at least one time sample.")

    @property
    def prediction_window_mask(self):
        """
        Computes the full prediction window mask by extending the base mask (which contains the prediction window
        mask for each time-lag) to all time-lags and low-pass filters.

        Returns:
            **prediction_window_mask** (*ndarray*) -- numpy 2-D boolean array of shape
            (``self.vector_dimensionality``, total_lags * ``self.vector_dimensionality``) containing the prediction
            window mask.
        """
        total_lags = len(self.time_lags) + self.num_low_pass_filters
        prediction_window_mask = np.tile(self._base_prediction_window_mask, (1, total_lags))
        return prediction_window_mask

    @property
    def predicted_components(self):
        """
        Finds the indices of the vector components for which any non-zero prediction weights have been assigned by
        the prediction window mask.

        Returns:
            **predicted_components** (*ndarray*) -- numpy 1-D integer array containing the indices of the predicted
            components.
        """
        predicted_components = np.sort(np.unique(self._base_prediction_window_mask.nonzero()[0]))
        return predicted_components

    @property
    def remaining_components(self):
        """
        Finds the indices of the vector components for which there are no prediction weights assigned by the
        prediction window mask. These components are not predicted, and instead are modeled purely as noise.

        Returns:
            **remaining_components** (*ndarray*) -- numpy 1-D array containing the indices of the remaining
            (non-predicted) vector components.
        """
        all_components = np.arange(self.vector_dimensionality)
        remaining_components = np.setdiff1d(all_components, self.predicted_components)
        return remaining_components

    @property
    def num_low_pass_filters(self):
        """
        Finds the number of low-pass filters used in the **Long-Range AR** model.

        Returns:
            **num_low_pass_filters** (*int*) -- the number of low-pass filters.
        """
        num_low_pass_filters = len(self.low_pass_filter_params)
        return num_low_pass_filters

    @property
    def num_prediction_weights(self):
        """
        Computes the number of (non-zero) prediction weights in the **Long-Range AR** model.

        Returns:
            **num_prediction_weights** (*int*) -- the number of (non-zero) prediction weights.
        """
        num_weights_per_lag = self._base_prediction_window_mask.sum()
        total_lags = len(self.time_lags) + self.num_low_pass_filters
        num_prediction_weights = num_weights_per_lag * total_lags
        return num_prediction_weights

    @property
    def num_parameters(self):
        """
        Calculates the number of parameters in the model. These parameters are estimated from training data and used
        to generate synthetic data.

        Returns:
            **total_num_params** (*int*) -- total number of parameters in the model.
        """
        total_num_params = (self.num_prediction_weights + self.vector_dimensionality * (self.vector_dimensionality + 1)
                            + self.num_low_pass_filters)

        return total_num_params

