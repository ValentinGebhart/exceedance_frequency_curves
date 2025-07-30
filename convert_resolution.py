from math import prod
from itertools import product

import numpy as np

from return_period_maps import ExceedanceCurve
from utils import round_to_array, prob_from_exceedance_frequency, exceedance_frequency_from_prob


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
    exceedance_frequencies = np.array(
        [
            exceedance_curves.exceedance_frequencies
            for exceedance_curves in exceedance_curves
        ]
    )
    # estimate probabilities from exceedance frequencies
    # TBD add warning!

    probabilities = prob_from_exceedance_frequency(
        exceedance_frequency=exceedance_frequencies, coincidence_fraction=coincidence_fraction)
    if value_resolution is None:
        value_resolution = np.nanmin(np.diff(values, axis=1))
    # round values to resolution
    value_bins = np.arange(
        np.nanmin(values), np.nanmax(values) + value_resolution, value_resolution
    )
    values = round_to_array(values, value_bins)

    # add zeros corresponding to nothing happens probability
    values = np.insert(values, 0, 0.0, axis=-1)

    # aggreagate values and probabilities
    aggregated_values = np.array(
        [aggregation_method(combination) for combination in product(*values)]
    )
    aggregated_probabilities = np.array(
        [prod(combination) for combination in product(*probabilities)]
    )

    # round aggregated values to resolution
    aggregated_bins = np.arange(
        np.nanmin(aggregated_values),
        np.nanmax(aggregated_values) + value_resolution,
        value_resolution,
    )
    aggregated_values = round_to_array(aggregated_values, aggregated_bins)
    unique_values, indices = np.unique(aggregated_values, return_inverse=True)
    indices = indices.reshape(aggregated_values.shape)
    corresponding_probabilities = np.array(
        [
            aggregated_probabilities[np.where(indices == index)].sum()
            for index in range(indices.max() + 1)
        ]
    )
    final_probabilities = np.zeros_like(aggregated_bins)
    for unique_value, corresponding_probabilitiy in zip(
        unique_values, corresponding_probabilities
    ):
        final_probabilities[np.where(aggregated_bins == unique_value)] = (
            corresponding_probabilitiy
        )
    final_exceedance_frequency = exceedance_frequency_from_prob(
       final_probabilities, coincidence_fraction=coincidence_fraction
    )
    # remove nothing happens bin
    aggregated_bins = aggregated_bins[1:]

    # frequency_range = np.where(
    #     (final_exceedance_frequency > exceedance_frequencies.min()) & 
    #     (final_exceedance_frequency < exceedance_frequencies.max())
    # )
    # aggregated_bins = aggregated_bins[frequency_range]
    # final_exceedance_frequency = final_exceedance_frequency[frequency_range]
    aggregated_return_period_curve = ExceedanceCurve(
        values=aggregated_bins,
        exceedance_frequencies=final_exceedance_frequency,
        time_unit=exceedance_curves[0].time_unit,
        value_unit=exceedance_curves[0].value_unit,
    )

    return aggregated_return_period_curve


def coarsen_resolution():
    return


def refine_resolution():
    return
