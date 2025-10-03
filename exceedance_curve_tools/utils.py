"""utils"""

import numpy as np
from scipy.stats import norm

### rounding functions
def round_to_array(obj, array):
    """
    Round the elements of `obj` to the nearest value in `arr`.

    Parameters:
    obj (array-like): The object containing values to be rounded.
    arr (array-like): The 1-D array of values to round to.

    Returns:
    np.ndarray: An array with the same shape as `obj`, where each
    element is replaced by the closest value from `arr`.
    """
    obj = np.asarray(obj)
    array = np.asarray(array)
    if array.shape != (array.size,):
        raise ValueError("arr must be a 1-D array.")

    # Find the index in array of the closest value for each value in object
    indices = np.abs(np.expand_dims(obj, -1) - array).argmin(axis=-1)

    # Replace each entry in object with the closest entry from array
    return array[indices]

### frequency/probability conversion functions
def prob_from_exceedance_frequency(exceedance_frequency, coincidence_fraction=1 / 12):
    """
    Convert exceedance frequency to probability.

    Parameters:
    exceedance_frequency (float or array-like): The exceedance frequency.
    time_unit (str): The time unit for the return period (default is "year").
    coincidence_fraction (float): Fraction of the year that the event coincides with
    (default is 1/12).

    Returns:
    float or np.ndarray: The probability corresponding to the exceedance frequency.
    """
    if not np.all(np.diff(exceedance_frequency, axis=-1) <= 0):
        raise ValueError(
            "Array must be sorted to convert from exceedance frequency to probability"
        )

    # compute probability of exceedance from exceedance frequency
    probs_exceedance = exceedance_probability_from_exceedance_frequency(
        exceedance_frequency, coincidence_fraction
    )
    # compute probabilities from exceedance probabilities
    probabilities = np.flip(
        np.diff(np.insert(np.flip(probs_exceedance, axis=-1), 0, 0.0, -1), axis=-1),
        axis=-1,
    )
    # include probability for nothing happening
    probabilities = np.insert(
        probabilities, 0, 1 - np.sum(probabilities, axis=-1), axis=-1
    )

    return probabilities


def exceedance_probability_from_exceedance_frequency(
    exceedance_frequency, coincidence_fraction=1 / 12
):
    """get exceedance probabilities from exceedance frequencies"""
    # convert time unit to coincidence window
    ex_freq = exceedance_frequency * coincidence_fraction
    # compute probability of exceedance from exceedance frequency
    return 1 - np.exp(-ex_freq)


def exceedance_frequency_from_exceedance_probability(
    exceedance_probability, coincidence_fraction=1 / 12
):
    """get exceedance frequencies from exceedance probabilities"""
    # recover exceedance frequencies
    ex_freq = -np.log(1 - exceedance_probability)
    # Undo the scaling by coincidence_fraction
    return ex_freq / coincidence_fraction


def exceedance_frequency_from_prob(probabilities, coincidence_fraction=1 / 12):
    """
    Inverse of prob_from_exceedance_frequency.
    Convert probabilities back to exceedance frequencies.

    Parameters
    ----------
    probabilities : float or array-like
        Probabilities as returned by prob_from_exceedance_frequency.
    coincidence_fraction : float, optional
        Fraction of the year that the event coincides with (default is 1/12).

    Returns
    -------
    float or np.ndarray
        The exceedance frequencies corresponding to the probabilities.
    """
    # Remove the probability for "nothing happening" (first entry)
    probs = np.delete(probabilities, 0, axis=-1)
    # Recover the exceedance probabilities from probabilities
    exceedance_probabilities = np.flip(
        np.cumsum(np.flip(probs, axis=-1), axis=-1), axis=-1
    )

    return exceedance_frequency_from_exceedance_probability(
        exceedance_probabilities, coincidence_fraction
    )


def frequency_from_exceedance_frequency(exceedance_frequency):
    """get frequencies from exceedance frequencies"""
    if not np.all(np.diff(exceedance_frequency, axis=-1) <= 0):
        raise ValueError(
            "Array must be sorted to convert from exceedance frequency to frequency"
        )

    return np.flip(
        np.diff(np.insert(np.flip(exceedance_frequency, axis=-1), 0, 0.0, -1), axis=-1),
        axis=-1,
    )

### sampling functions
def get_correlated_quantiles(d, correlation_factor, n_samples):
    """sample correlated quantiles"""
    # create covariance matrix
    correlation_factor = float(correlation_factor)
    if d == 1 or correlation_factor == 0:
        return np.random.random(size=(n_samples, d))

    # check if given values lead to positive definite covariance matrix
    if correlation_factor > 1 or correlation_factor < -1:
        raise ValueError("Correlation factor must be between -1 and 1.")

    # correct correlation_factor to make covariance matrix positive semidefinite.
    if correlation_factor < 0:
        correlation_factor *= 1 / (d - 1)

    cov_matrix = np.full((d, d), correlation_factor)
    cov_matrix += np.diag(np.full(d, 1.0 - correlation_factor))

    # sample from multivariate normal distribtuion
    mean = np.zeros(d)
    normal_samples = np.random.multivariate_normal(mean, cov_matrix, size=n_samples)

    # transform normal samples to uniform(0, 1)
    return norm.cdf(normal_samples)

# def check_if_corr_valid_given_marginals(p, corr):
#     """check if given correlation matrix is valid given marginal probabilities"""
#     n = len(p)

#     # Compute thresholds for each variable
#     thresholds = norm.ppf(1 - p)

#     # Compute feasible correlation bounds
#     lower_bounds = np.zeros((n, n))
#     upper_bounds = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 lower_bounds[i, j] = 1.0
#                 upper_bounds[i, j] = 1.0
#             else:
#                 lower_bounds[i, j] = (
#                     (np.exp(-thresholds[i] - thresholds[j]) - (1 - p[i]) * (1 - p[j]))
#                     / np.sqrt(p[i] * (1 - p[i]) * p[j] * (1 - p[j]))
#                 )
#                 upper_bounds[i, j] = (
#                     (min(p[i], p[j]) - (1 - p[i]) * (1 - p[j]))
#                     / np.sqrt(p[i] * (1 - p[i]) * p[j] * (1 - p[j]))
#                 )

#     # Check if all correlations are within bounds
#     if np.all((corr >= lower_bounds) & (corr <= upper_bounds)):
#         return True
#     else:
#         return False

# def sample_correlated_bernoulli(p, corr, n_samples):
#     """
#     Sample n Bernoulli variables with given means and target correlation matrix using a Gaussian copula.

#     Parameters
#     ----------
#     p : array-like, shape (n,)
#         Marginal probabilities for each variable.
#     corr : array-like, shape (n, n)
#         Target correlation matrix (must be symmetric, positive semi-definite, 1 on diagonal).
#     n_samples : int
#         Number of samples to generate.

#     Returns
#     -------
#     samples : ndarray, shape (n_samples, n)
#         Sampled Bernoulli variables (0 or 1).
#     """
#     p = np.asarray(p)
#     corr = np.asarray(corr)
#     n = len(p)

#     # Check correlation matrix
#     if corr.shape != (n, n):
#         raise ValueError("corr must be of shape (n, n)")
#     if not np.allclose(np.diag(corr), 1):
#         raise ValueError("Diagonal of corr must be 1")
#     if not np.allclose(corr, corr.T):
#         raise ValueError("corr must be symmetric")

#     # Compute thresholds for each variable
#     thresholds = norm.ppf(1 - p)

#     # Sample from multivariate normal
#     mean = np.zeros(n)
#     mvn_samples = np.random.multivariate_normal(mean, corr, size=n_samples)

#     # Apply thresholds to get Bernoulli samples
#     samples = (mvn_samples < thresholds).astype(int)
#     return samples

### other utils
def fill_edges(a):
    """fill initial (final) NaN values along axis 1 with first (final)
    non-NaN value. NaNs in between non-NaNs are not filled."""
    not_nan = ~np.isnan(a)
    if not np.any(not_nan):
        return a  # all NaN, nothing to fill
    first = np.argmax(not_nan)
    last = len(a) - np.argmax(not_nan[::-1]) - 1
    a[:first] = a[first]
    a[last + 1 :] = a[last]
    return a
