import numpy as np

# Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2026-1309.

def find_top_principal_components(pc_variances, percent_variance):
    """
    Finds the top principal components containing the given percentage of the total variance.

    Args:
        pc_variances (ndarray): numpy 1-D array containing the variance of each principal component.
        percent_variance (float): percentage of the total variance to look for.

    Returns:
        **num_components** (*int*) -- the number of top principal components containing the given percentage of the
        total variance.
    """
    assert ((percent_variance > 0) and (percent_variance <= 1.0))
    assert ((pc_variances.ndim == 1) and (pc_variances >= 0).all())
    assert np.all(np.diff(pc_variances) <= 1e-14)    # Ensure that the variances are in descending order

    total_variance = pc_variances.sum()
    threshold_variance = total_variance * percent_variance

    # Find the first principal component for which the cumulative sum is at least the given percentage of the total sum:
    cumulative_variance = np.cumsum(pc_variances)
    threshold_variance_index = int(np.searchsorted(cumulative_variance, threshold_variance, side='left'))

    # Handle the case that you need all principal components to cover the given percent_variance:
    num_components = min(threshold_variance_index + 1, len(pc_variances))

    return num_components



def compute_pca(data):
    """
    Computes the principal components and their associated variances for an input array containing samples of a
    multivariate Gaussian distribution. Takes the Singular Value Decomposition (SVD) of the covariance matrix.

    Args:
        data (ndarray): numpy 2-D array of shape (vector dimensionality, number of samples) containing samples
            of the multivariable Gaussian distribution.

    Returns:
        - **data_mean** (*ndarray*) -- numpy 1-D array of shape (vector dimensionality,) containing the sample mean
          vector (i.e., the sample mean of each vector component).
        - **principal_components** (*ndarray*) -- numpy 2-D array of shape (vector dimensionality, vector
          dimensionality) containing the principal component matrix. The principal components are the columns of this
          matrix.
        - **pc_variances** (*ndarray*) -- numpy 1-D array of shape (vector dimensionality,) containing the variance of
          each principal component.
    """
    assert (data.ndim == 2)

    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values, which are not supported.")

    # Compute and remove mean vector
    data_mean = np.average(data, axis=1)
    data_mean_removed = data - data_mean[:, np.newaxis]

    # Estimates the covariance matrix of the distribution:
    covariance_estimate = np.dot(data_mean_removed, data_mean_removed.T) / data_mean_removed.shape[1]

    # Compute SVD to find principal components and their variances:
    principal_components, pc_variances = np.linalg.svd(covariance_estimate)[:2]

    return data_mean, principal_components, pc_variances


def generative_pca_algorithm(num_samples, covariance_modulation_matrix, mean_vector=None):
    """
    Generates samples from a multivariate Gaussian distribution. This uses the PCA generative algorithm, which
    generates white noise vectors and then (1) multiplies them by a modulation matrix to set the covariance matrix and
    (2) adds the mean vectors.

    Args:
        num_samples (int): number of samples to generate.
        covariance_modulation_matrix (ndarray): numpy 2-D array of shape (random vector dimensionality, random vector
            dimensionality) containing a matrix which scales unit-variance white noise to have the desired spatial
            covariance matrix. This matrix is determined by the matrices of principal components their variances.
        mean_vector (ndarray, optional): [Default=None] numpy 1-D array of shape (random vector dimensionality,)
            containing the mean of the distribution.

            - If set to None, a mean vector of zero is used.

    Returns:
        **samples** (*ndarray*) -- 2-D array of shape (random vector dimensionality, num_samples) whose columns contain
        samples from the desired multivariate Gaussian distribution.
    """
    assert (num_samples > 0)
    assert (covariance_modulation_matrix.ndim == 2)
    assert (covariance_modulation_matrix.shape[0] == covariance_modulation_matrix.shape[1])

    if mean_vector is not None:
        assert ((mean_vector.ndim == 1) and (mean_vector.shape[0] == covariance_modulation_matrix.shape[0]))
    else:
        mean_vector = np.zeros(covariance_modulation_matrix.shape[0])

    # Gaussian i.i.d random variables (mean 0, variance 1):
    white_noise = np.random.normal(size=(covariance_modulation_matrix.shape[0], num_samples))

    # Samples from the distribution
    samples = np.dot(covariance_modulation_matrix, white_noise) + mean_vector[:, np.newaxis]

    return samples
