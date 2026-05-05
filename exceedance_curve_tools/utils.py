"""utils"""

import numpy as np
from scipy.stats import norm

###########################
## frequency probability ##
## conversion functions ###
###########################

# conversion goes along
# prob (sorted) <-> ex_prob <-> ex_freq <-> freq (sorted)
# to convert ex_prob <-> ex_freq a time window must be given (by coincidence fraction)


def ex_prob_from_ex_freq(exceedance_frequency, coincidence_fraction=1 / 12):
    """get exceedance probabilities from exceedance frequencies"""
    # convert time unit to coincidence window
    ex_freq = exceedance_frequency * coincidence_fraction
    # compute probability of exceedance from exceedance frequency
    return 1 - np.exp(-ex_freq)


def ex_freq_from_ex_prob(exceedance_probability, coincidence_fraction=1 / 12):
    """get exceedance frequencies from exceedance probabilities"""
    # recover exceedance frequencies
    ex_freq = -np.log(1 - exceedance_probability)
    # Undo the scaling by coincidence_fraction
    return ex_freq / coincidence_fraction


def freq_from_ex_freq(exceedance_frequency):
    """get frequencies from exceedance frequencies"""
    if not np.all(np.diff(exceedance_frequency, axis=-1) <= 0):
        raise ValueError(
            "Array must be sorted to convert from exceedance frequency to frequency"
        )

    return np.flip(
        np.diff(np.insert(np.flip(exceedance_frequency, axis=-1), 0, 0.0, -1), axis=-1),
        axis=-1,
    )


def ex_freq_from_freq(frequency):
    """get exceedance frequencies from frequencies"""
    return np.flip(np.cumsum(np.flip(frequency, axis=-1), axis=-1), axis=-1)


def prob_from_ex_prob(exceedance_probability):
    """get probabilities from exceedance probabilities"""
    if not np.all(np.diff(exceedance_probability, axis=-1) <= 0):
        raise ValueError(
            "Array must be sorted to convert from exceedance probability to probability"
        )

    # compute probabilities from exceedance probabilities
    probabilities = np.flip(
        np.diff(
            np.insert(np.flip(exceedance_probability, axis=-1), 0, 0.0, -1), axis=-1
        ),
        axis=-1,
    )
    # include probability for nothing happening
    probabilities = np.insert(
        probabilities, 0, 1 - np.sum(probabilities, axis=-1), axis=-1
    )

    return probabilities


def ex_prob_from_prob(probabilities):
    """get exceedance probabilities from probabilities"""
    # Remove the probability for "nothing happening" (first entry)
    probs = np.delete(probabilities, 0, axis=-1)
    # Recover the exceedance probabilities from probabilities
    return np.flip(np.cumsum(np.flip(probs, axis=-1), axis=-1), axis=-1)


def prob_from_ex_freq(exceedance_frequency, coincidence_fraction=1 / 12):
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

    # compute probability of exceedance from exceedance frequency
    exceedance_probability = ex_prob_from_ex_freq(
        exceedance_frequency, coincidence_fraction
    )
    return prob_from_ex_prob(exceedance_probability)


def ex_freq_from_prob(probabilities, coincidence_fraction=1 / 12):
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
    exceedance_probabilities = ex_prob_from_prob(probabilities)
    return ex_freq_from_ex_prob(exceedance_probabilities, coincidence_fraction)


##########################
### rounding functions ###
##########################


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


##########################
### sampling functions ###
##########################


def get_correlated_quantiles(d, correlation_factor, n_samples, rng=None):
    """sample correlated quantiles"""
    if rng is None:
        rng = np.random.default_rng()

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
    normal_samples = rng.multivariate_normal(mean, cov_matrix, size=n_samples)

    # transform normal samples to uniform(0, 1)
    return norm.cdf(normal_samples)


##########################
####### other utils ######
##########################


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


def aggregation_period(time_unit, aggregation_time_fraction):
    """naming of different aggreagtion period"""
    if time_unit == "year":
        if aggregation_time_fraction == 1:
            return "year"
        elif aggregation_time_fraction == 1 / 12:
            return "month"
        elif aggregation_time_fraction == 1 / 365:
            return "day"
        elif aggregation_time_fraction == 10:
            return "decade"
    return f"{aggregation_time_fraction:.2g} {time_unit}"
