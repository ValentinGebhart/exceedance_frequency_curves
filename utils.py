import numpy as np
from scipy.stats import norm


def round_to_array(obj, array):
    """
    Round the elements of `obj` to the nearest value in `arr`.

    Parameters:
    obj (array-like): The object containing values to be rounded.
    arr (array-like): The 1-D array of values to round to.

    Returns:
    np.ndarray: An array with the same shape as `obj`, where each element is replaced by the closest value from `arr`.
    """
    obj = np.asarray(obj)
    array = np.asarray(array)
    if array.shape != (array.size,):
        raise ValueError("arr must be a 1-D array.")

    # Find the index in array of the closest value for each value in object
    indices = np.abs(np.expand_dims(obj, -1) - array).argmin(axis=-1)

    # Replace each entry in object with the closest entry from array
    return array[indices]


def prob_from_exceedance_frequency(exceedance_frequency, coincidence_fraction=1 / 12):
    """
    Convert exceedance frequency to probability.

    Parameters:
    exceedance_frequency (float or array-like): The exceedance frequency.
    time_unit (str): The time unit for the return period (default is "year").
    coincidence_fraction (float): Fraction of the year that the event coincides with (default is 1/12).

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
    )  # probabilities = np.concatenate([np.expand_dims(1- np.sum(probabilities, axis=-1), axis=-1), probabilities])
    # include probability for nothing happening
    probabilities = np.insert(
        probabilities, 0, 1 - np.sum(probabilities, axis=-1), axis=-1
    )

    return probabilities


def exceedance_probability_from_exceedance_frequency(
    exceedance_frequency, coincidence_fraction=1 / 12
):
    # convert time unit to coincidence window
    ex_freq = exceedance_frequency * coincidence_fraction
    # compute probability of exceedance from exceedance frequency
    return 1 - np.exp(-ex_freq)


def exceedance_frequency_from_exceedance_probability(
    exceedance_probability, coincidence_fraction=1 / 12
):
    # recover exceedance frequencies
    ex_freq = -np.log(1 - exceedance_probability)
    # Undo the scaling by coincidence_fraction
    return ex_freq / coincidence_fraction


def exceedance_frequency_from_prob(probabilities, coincidence_fraction=1 / 12):
    """
    Inverse of prob_from_exceedance_frequency.
    Convert probabilities back to exceedance frequencies.

    Parameters:
    probabilities (float or array-like): Probabilities as returned by prob_from_exceedance_frequency.
    coincidence_fraction (float): Fraction of the year that the event coincides with (default is 1/12).

    Returns:
    float or np.ndarray: The exceedance frequencies corresponding to the probabilities.
    """
    # Remove the probability for "nothing happening" (first entry)
    probs = np.delete(probabilities, 0, axis=-1)
    # Recover the exceedance probabilities from probabilities
    exceedance_probabilities = np.flip(
        np.cumsum(np.flip(probs, axis=-1), axis=-1), axis=-1
    )
    # recover exceedance frequencies
    ex_freq = -np.log(1 - exceedance_probabilities)
    # Undo the scaling by coincidence_fraction
    exceedance_frequency = ex_freq / coincidence_fraction
    return exceedance_frequency


def frequency_from_exceedance_frequency(exceedance_frequency):
    if not np.all(np.diff(exceedance_frequency, axis=-1) <= 0):
        raise ValueError(
            "Array must be sorted to convert from exceedance frequency to frequency"
        )

    return np.flip(
        np.diff(np.insert(np.flip(exceedance_frequency, axis=-1), 0, 0.0, -1), axis=-1),
        axis=-1,
    )


def get_correlated_quantiles(d, correlation_factor, n_samples):
    # create covariance matrix
    if d == 1 or correlation_factor == 0:
        return np.random.random(size=(n_samples, d))

    cov_matrix = np.full((d, d), correlation_factor)
    cov_matrix += np.diag(np.full(d, 1.0 - correlation_factor))

    # check if given values lead to positive definite covariance matrix
    if correlation_factor > 1 or correlation_factor < -1 / (d - 1):
        raise ValueError(
            f"Given parameters (correlation factor {correlation_factor} and dimension {d}) result"
            f" in non positive definite covariance matrix for sampling. The values must fulfill"
            "correlation_factor >= -1 / (d - 1)."
        )

    # sample from multivariate normal distribtuion
    mean = np.zeros(d)
    normal_samples = np.random.multivariate_normal(mean, cov_matrix, size=n_samples)

    # transform normal samples to uniform(0, 1)
    return norm.cdf(normal_samples)


def sort_two_arrays_by_first(arr1, arr2, ascending=True):
    if ascending:
        order = np.argsort(arr1)
    else:
        order = np.argsort(arr1)[::-1]
    return arr1[order], arr2[order]

def fill_edges(a):
    not_nan = ~np.isnan(a)
    if not np.any(not_nan):
        return a  # all NaN, nothing to fill
    first = np.argmax(not_nan)
    last = len(a) - np.argmax(not_nan[::-1]) - 1
    a[:first] = a[first]
    a[last+1:] = a[last]
    return a