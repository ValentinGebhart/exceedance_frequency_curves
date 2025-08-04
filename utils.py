import numpy as np


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
    # convert time unit to coincidence window
    ex_freq = exceedance_frequency * coincidence_fraction
    # compute probability of exceedance from exceedance frequency
    probs_exceedance = 1 - np.exp(-ex_freq)
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
    return np.flip(
        np.diff(np.insert(np.flip(exceedance_frequency, axis=-1), 0, 0.0, -1), axis=-1),
        axis=-1,
    )
