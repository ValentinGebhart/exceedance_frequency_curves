import numpy as np
from math import prod
from itertools import product

from return_period_maps import ExceedanceCurve
from utils import round_to_array, prob_to_freq, freq_to_prob


def combine_exceedance_curves(
    exceedance_curves,
    value_resolution=None,
    aggregation_method=sum,
    coincidence_fraction=1 / 12,
):
    """_summary_

    Parameters
    ----------
    exceedance_curves : _type_
        _description_
    value_resolution : _type_, optional
        _description_, by default None
    aggregation_method : _type_, optional
        _description_, by default sum
        note handling of  0

    Returns
    -------
    _type_
        _description_
    """

    values = np.array(
        [return_period_curve.values for return_period_curve in exceedance_curves]
    )
    frequencies = np.array(
        [
            exceedance_curves.exceedance_frequencies
            for exceedance_curves in exceedance_curves
        ]
    )

    # convert from exceedance frequencies to frequencies
    frequencies = np.flip(
        np.diff(np.insert(np.flip(frequencies, axis=1), 0, 0.0, axis=1), axis=1), axis=1
    )
    # convert to probabilities
    # probabilities = prob_to_freq(frequencies, coincidence_fraction=1/12)

    if value_resolution is None:
        value_resolution = np.diff(values, axis=1)

    # round values to resolution
    value_bins = np.arange(
        values.min(), values.max() + value_resolution, value_resolution
    )
    values = round_to_array(values, value_bins)

    # add zeros such that also only values for one list can contribute
    values = np.insert(values, 0, 0.0, axis=1)
    zero_frequency = (
        1 - frequencies.max(axis=1) * coincidence_fraction
    ) / coincidence_fraction

    frequencies = np.insert(frequencies, 0, zero_frequency, axis=1)
    print(frequencies)
    # probabilities = np.insert(probabilities, 0, 1., axis=1)

    # aggreagate values and probabilities
    aggregated_values = np.array(
        [aggregation_method(combination) for combination in product(*values)]
    )
    aggregated_frequencies = np.array(
        [prod(combination) for combination in product(*frequencies)]
    )
    aggregated_frequencies *= coincidence_fraction ** (len(exceedance_curves) - 1)

    # round aggregated values to resolution
    aggregated_bins = np.arange(
        aggregated_values[aggregated_values != 0].min(),
        aggregated_values.max() + value_resolution,
        value_resolution,
    )
    aggregated_values = round_to_array(aggregated_values, aggregated_bins)
    unique_values, indices = np.unique(aggregated_values, return_inverse=True)
    indices = indices.reshape(aggregated_values.shape)
    corresponding_frequencies = np.array(
        [
            aggregated_frequencies[np.where(indices == index)].sum()
            for index in range(indices.max() + 1)
        ]
    )
    final_frequencies = np.zeros_like(aggregated_bins)
    for unique_value, corresponding_frequency in zip(
        unique_values, corresponding_frequencies
    ):
        final_frequencies[np.where(aggregated_bins == unique_value)] = (
            corresponding_frequency
        )
    # final_frequencies = prob_to_freq(final_probabilities, coincidence_fraction=1/12)
    aggregated_return_period_curve = ExceedanceCurve(
        values=aggregated_bins,
        exceedance_frequencies=np.cumsum(final_frequencies[::-1])[::-1],
        time_unit=exceedance_curves[0].time_unit,
        value_unit=exceedance_curves[0].value_unit,
    )

    return aggregated_return_period_curve


def coarsen_resolution():
    return


def refine_resolution():
    return
