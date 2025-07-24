import numpy as np
from itertools import product

from return_period_maps import ExceedanceCurve
from utils import round_to_array


def combine_exceedance_curves(
    exceedance_curves,
    value_resolution=None,
    aggregation_method=sum,
):
    values = np.array(
        [return_period_curve.values for return_period_curve in exceedance_curves]
    )
    frequencies = np.array(
        [
            exceedance_curves.exceedance_frequencies
            for exceedance_curves in exceedance_curves
        ]
    )

    if value_resolution is None:
        value_resolution = np.diff(values, axis=1)

    # round values to resolution
    value_bins = np.arange(
        values.min(), values.max() + value_resolution, value_resolution
    )
    values = round_to_array(values, value_bins)

    # aggreagate values and frequencies
    aggregated_values = np.array(
        [aggregation_method(combination) for combination in product(*values)]
    )
    aggregated_frequencies = np.array(
        [sum(combination) for combination in product(*frequencies)]
    )

    # round aggregated values to resolution
    aggregated_bins = np.arange(
        aggregated_values.min(),
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

    aggregated_return_period_curve = ExceedanceCurve(
        values=aggregated_bins,
        exceedance_frequencies=final_frequencies,
        time_unit=exceedance_curves[0].time_unit,
        value_unit=exceedance_curves[0].value_unit,
    )

    return aggregated_return_period_curve


def coarsen_resolution():
    return


def refine_resolution():
    return
