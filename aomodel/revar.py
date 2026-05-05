import numpy as np
import aomodel.utils as utils
import aomodel.pca as pca
from aomodel.long_range_ar import LongRangeAR
import time
from datetime import timedelta
import os
import warnings

# Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2026-1309.

class ReVAR(LongRangeAR):
    """
    Implements the **Re-whitened Vector AutoRegression (ReVAR)** algorithm to generate synthetic time series of images
    with the same spatial and temporal statistics as an input data set. For a thorough description of this model, see
    :ref:`Theory`.

    Args:
        data_mask (ndarray, optional): [Default=None] numpy 2-D boolean array of shape (image height, image width)
            indicating which 2-D indices of each time-step correspond to valid pixel values.

            - If set to None, then ``load_file`` must be provided. In this case, the class loads the mask from the file.

        time_lags (Union[int, list, ndarray], optional): [Default=1] either an integer, list, or numpy 1-D array
            indicating the time lags to use for the model.

            - If ``time_lags`` is an integer, the function uses the previous time-steps up to this integer value.

            - If ``time_lags`` is a list or numpy 1-D array, the model uses these time lags.

        num_low_pass_filters (int, optional): [Default=0] the number of low-pass filters to use in the linear time
            predictive model.

        load_file (str, optional): [Default=None] directory to a file from which the model's instance variables can
            be loaded.

        prediction_window_structure (PredictionWindowStructure, optional): [Default=None] an instance of a sub-class
                of ``PredictionWindowStructure`` implementing get_mask(). Passed as input to the constructor of
                ``LongRangeAR``.

                - If set to None, the method builds a prediction window structure from `**kwargs`.

        **kwargs (dict, optional): Keyword arguments passed to ``LongRangeAR``.

            - **prediction_subspace_dimension** (*int*) -- number of (top) vector components to use for linear time
              prediction.

            - **low_pass_filter_params** (*ndarray*) -- numpy 1-D array of length ``num_low_pass_filters`` containing
              the parameters of each low-pass filter.
    """

    def __init__(self, data_mask=None, time_lags=1, num_low_pass_filters=0, load_file=None,
                 prediction_window_structure=None, **kwargs):
        if data_mask is None:
            assert (load_file is not None)
            num_pixels = None
        else:
            assert (data_mask.dtype == bool)
            assert (data_mask.ndim == 2)
            num_pixels = data_mask.sum()

        # Initializes instance variables:
        self.data_mask = data_mask
        self.principal_components = None
        self.pc_variances = None
        self.standard_deviation_vector = None
        self.mean_vector = None

        print("\nInitializing ReVAR model...")
        super().__init__(vector_dimensionality=num_pixels, time_lags=time_lags,
                         num_low_pass_filters=num_low_pass_filters, load_file=load_file,
                         prediction_window_structure=prediction_window_structure, print_statements=False, **kwargs)

        if load_file is None:
            print("ReVAR model initialized. Use ReVAR.fit() to fit this model to data or ReVAR.load() to load a "
                  "pre-trained model.\n")


    def fit(self, training_data, percent_variance=None, cutoff_frequency=None, tps_block_size=None):
        """
        Estimates the parameters of **ReVAR** from training data. Uses ``LongRangeAR.fit`` to compute the prediction
        weights, low-pass filter parameters (if used), and noise modulation matrix.

        Args:
            training_data (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing
                the data to fit.

            percent_variance (float, optional): [Default=None] percent variance of the subspace to use for linear time
                prediction in the **Long-Range AR** model.

                - If set to None, linear prediction is applied to all principal coefficients.

            cutoff_frequency (float, optional): [Default=None] cutoff frequency (in units of cycles/sample) to use for
                estimation of the low-pass filter parameters.

                - If set to None and low-pass filters are used, this method estimates the cutoff frequency as the peak
                  frequency of the temporal power spectrum (TPS) of the training data.

            tps_block_size (int, optional): [Default=None] time-block size for calculating the TPS of the principal
                coefficients.

                - If set to None and the low-pass filter parameters need to be estimated, the function finds the block
                  size so that the TPS estimates are averaged over 100 blocks. This ensures that the signal-to-noise
                  ratio of each 1-D TPS estimate is at least 10.
        """
        self.__validate_training_data(training_data=training_data)

        # Warns about overwriting previously-fit models:
        if self._is_fitted:
            warnings.warn("This model instance has already been fitted. Calling fit() will overwrite existing "
                          "parameters.", UserWarning)

        print("\nReVAR Parameter Estimation\n"
              "==========================\n"
              "Number of time-steps in training data:  %d\n" % training_data.shape[0])
        start_time = time.time()

        # Run ReVAR pre-processing to normalize the data and compute a spatial PCA:
        principal_coefficients = self.pre_processing(training_data=training_data,
                                                     percent_variance=percent_variance)

        print("> Long-Range AR model fitting started...")
        # First the Long-Range AR model to the principal coefficients:
        super().fit(training_data=principal_coefficients,
                    cutoff_frequency=cutoff_frequency,
                    tps_block_size=tps_block_size,
                    print_statements=False)
        print("> Long-Range AR model fitting completed.\n")

        runtime_in_seconds = time.time() - start_time
        elapsed_time = str(timedelta(seconds=runtime_in_seconds))
        print("==========================")
        print(f"ReVAR Parameter Estimation completed in {elapsed_time} (hr:min:sec)\n")

        # Total number of parameters estimated from data:
        print(f"Total number of parameters: ", self.num_parameters)

    def run(self, num_time_steps, initial_vectors=None, print_statements=False):
        """
        Runs the **ReVAR** data synthesis algorithm using estimated parameters from training data, as set by either the
        ``self.fit`` or ``self.load`` instance methods. This model generates synthetic data with the same statistics as
        training data.

        Args:
            num_time_steps (int): the number of time-steps of synthetic data to generate.
            initial_vectors (ndarray, optional): [Default=None] numpy 2-D array of shape (data vector dimensionality,
                number of time lags) containing the initial data vectors used by the data synthesis algorithm.

                - If set to None, generates initial vectors that have the same spatial distribution as the principal
                  coefficients of training data.

            print_statements (str, optional): [Default=False] whether to print out ``LongRangeAR`` statements.

        Returns:
            **synthetic_data** (*ndarray*) -- numpy 3-D array of shape (``num_time_steps``, image height, image width)
            containing the model's output synthetic data.

        Raises:
            RuntimeError: if the attribute ``self._if_fitted`` is not True (indicating that the model has not been fit).
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Please call 'fit()' or 'load()' before running generation.")

        print("\nReVAR Data Synthesis\n"
              "====================\n"
              f"Generating {num_time_steps} time-steps of synthetic data...\n")
        start_time = time.time()

        # Uses PCA to generate the initial vectors (which have the same spatial distribution as the principal
        # coefficients of training data):
        if initial_vectors is not None:
            assert (initial_vectors.shape == (self.vector_dimensionality, max(self.time_lags)))
        else:
            initial_white_noise = np.random.normal(size=(self.principal_components.shape[0], max(self.time_lags)))
            initial_vectors = np.multiply(np.sqrt(self.pc_variances)[:, None], initial_white_noise,
                                          out=initial_white_noise)

        # Generates synthetic data using the Long-Range AR model:
        synthetic_data = super().run(num_time_steps=num_time_steps,
                                     initial_vectors=initial_vectors,
                                     print_statements=False)

        # Projects the output of the AR model back onto the standard coordinate space:
        synthetic_data = np.dot(self.principal_components, synthetic_data)

        # Puts the data in the correct units:
        synthetic_data = (np.multiply(synthetic_data, self.standard_deviation_vector[:, np.newaxis]) +
                          self.mean_vector[:, np.newaxis])

        # Switch from vectors to images:
        synthetic_data = utils.vec_to_img(data_vec=synthetic_data.T, mask=self.data_mask)

        runtime_in_seconds = time.time() - start_time
        elapsed_time = str(timedelta(seconds=runtime_in_seconds))
        print("ReVAR Data Synthesis completed in {} (hr:min:sec)\n".format(elapsed_time))

        return synthetic_data

    def save(self, save_file):
        """
        Saves all necessary information to re-construct the trained ``ReVAR`` model with a new instance.

        Args:
            save_file (str): directory of the file to which the data will be saved
        """
        assert (type(save_file) == str)
        print(f"Saving ReVAR model to {os.path.abspath(save_file)}...")
        valid_indices_of_prediction_weights_array = (self.prediction_window_indices != -1)
        prediction_weights = self.prediction_weights[valid_indices_of_prediction_weights_array]

        save_arrays = {'prediction_window_mask': self._base_prediction_window_mask,
                       'noise_modulation': self.noise_modulation, 'residuals_mean': self.residuals_mean,
                       'prediction_weights': prediction_weights, 'time_lags': self.time_lags,
                       'principal_components': self.principal_components, 'pc_variances': self.pc_variances,
                       'standard_deviation_vector': self.standard_deviation_vector, 'mean_vector': self.mean_vector,
                       'data_mask': self.data_mask}

        if self.prediction_subspace_dimension is not None:
            save_arrays['prediction_subspace_dimension'] = self.prediction_subspace_dimension

        if self.low_pass_filter_params is not None:
            save_arrays['low_pass_filter_params'] = self.low_pass_filter_params

        np.savez(save_file, **save_arrays)
        print(f"ReVAR model saved to {os.path.abspath(save_file)}.\n")

    def load(self, load_file, print_statements=False):
        """
        Loads the ``ReVAR`` model information as saved by the ``self.save`` method and re-constructs the model.

        Args:
            load_file (str): directory of the file from which the data will be loaded
            print_statements (str, optional): [Default=False] whether to print out ``LongRangeAR`` loading statements.
        """
        assert (type(load_file) == str)
        print(f"Loading ReVAR model from {os.path.abspath(load_file)}...")
        super().load(load_file=load_file, print_statements=print_statements)

        # Load data containing instance variables:
        data = np.load(file=load_file, allow_pickle=True)

        # Ensure that the loaded model is compatible with the current model:
        self.data_mask = data['data_mask']

        # Saves PCA instance variables:
        self.principal_components = data['principal_components']
        self.pc_variances = data['pc_variances']
        self.standard_deviation_vector = data['standard_deviation_vector']
        self.mean_vector = data['mean_vector']

        print(f"Finished loading ReVAR model from {os.path.abspath(load_file)}.")

        # Total number of parameters estimated from data:
        print(f"Total number of parameters: ", self.num_parameters, '\n')

    def pre_processing(self, training_data, percent_variance=None):
        """
        Pre-processing step of **ReVAR** parameter estimation: i) normalize the training data by its sample mean and
        standard deviation vectors and ii) compute a spatial PCA from the normalized training data.

        Args:
            training_data (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing
                the data to fit.

            percent_variance (float, optional): [Default=None] percent variance of the subspace to use for linear time
                prediction in the ``LongRangeAR`` model.

                - If set to None, linear prediction is applied to all principal coefficients.

        Returns:
            **principal_coefficients** (*ndarray*) -- numpy 2-D array of shape (vector dimensionality, number of images)
            containing the principal coefficient vectors of the (normalized) training data.
        """
        assert (training_data.ndim == 3)

        print("> ReVAR pre-processing started...")
        # Data normalization: remove mean and standard deviation vectors
        data_matrix = utils.img_to_vec(image_data=training_data, mask=self.data_mask).T
        self.mean_vector = np.average(data_matrix, axis=1)
        data_mean_removed = data_matrix - self.mean_vector[:, np.newaxis]
        self.standard_deviation_vector = np.sqrt(np.sum(data_mean_removed ** 2, axis=1) / training_data.shape[0])

        # If there are (temporally) constant components, allow normalization:
        with np.errstate(divide='ignore', invalid='ignore'):
            data_normalized = data_mean_removed / self.standard_deviation_vector[:, np.newaxis]

        # Handle the case of zero variance:
        data_normalized = np.nan_to_num(data_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        # Take a spatial PCA of the data:
        self.principal_components, self.pc_variances = pca.compute_pca(data=data_normalized)[1:]

        # Compute the principal coefficients:
        principal_coefficients = np.dot(self.principal_components.T, data_normalized)

        # Calculates the number of components to include in the subspace:
        if percent_variance is not None:
            num_coefficients = pca.find_top_principal_components(self.pc_variances, percent_variance)

            # Re-sets the prediction window indices of the linear time prediction model:
            self.create_model_structure(prediction_subspace_dimension=num_coefficients)

        print("> ReVAR pre-processing completed.\n")

        return principal_coefficients

    def __validate_training_data(self, training_data):
        """
        Ensures that the training data is compatible with the ``ReVAR`` class and the data mask of the model instance.

        Args:
            training_data (ndarray): numpy 3-D array of shape (number of images, image height, image width) containing
                the data to fit.

        Raises:
            ValueError: if the training data is not compatible with the class or with ``self.data_mask``.
        """
        if training_data.ndim != 3:
            raise ValueError(f"Training data must be 2-dimensional, got shape {training_data.shape}.")

        # Make sure the training data images have the same shape as self.data_mask:
        if (training_data.shape[1], training_data.shape[2]) != self.data_mask.shape:
            raise ValueError(f"Training data images must have the same shape as self.data_mask,"
                             f" got shape {(training_data.shape[1], training_data.shape[2])}.")

        if training_data.shape[0] == 0:
            raise ValueError(f"Training data vector must have at least one time sample.")

        # Ensure that the training data has no values of "nan" inside of self.data_mask:
        mask_compatibility = np.isnan(utils.img_to_vec(image_data=training_data, mask=self.data_mask))
        if mask_compatibility.any():
            num_incompatible_components = np.count_nonzero(mask_compatibility)
            raise ValueError(f"Training data is incompatible with self.data_mask, "
                             f"found {num_incompatible_components} incompatible values in data.")

    @property
    def num_parameters(self):
        """
        Calculates the number of parameters in the model. These parameters are estimated from training data and used
        to generate synthetic data.

        Returns:
            **total_num_params** (*int*) -- total number of parameters in the model.
        """
        num_pixels = self.data_mask.sum()
        total_num_params = self.num_prediction_weights + 2 * num_pixels * (num_pixels + 2) + self.num_low_pass_filters

        return total_num_params
