"""
ExceedanceCurve class and corresponding functions
"""

from math import prod
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    round_to_array,
    prob_from_exceedance_frequency,
    exceedance_frequency_from_prob,
)


class ExceedanceCurve:
    """_summary_"""

    def __init__(self, values, exceedance_frequencies, time_unit=None, value_unit=None):

        if len(values) != len(exceedance_frequencies):
            raise ValueError(
                f"Number of threshold values {len(values)} different to number of exceedance frequencies {len(exceedance_frequencies)}"
            )
        self.values = values
        self.exceedance_frequencies = exceedance_frequencies
        self.time_unit = time_unit if time_unit is not None else "year"
        self.value_unit = value_unit if value_unit is not None else "USD"

    def average_annual_impact(self):
        if self.time_unit != "year":
            raise ValueError(
                "Time unit must year 'year' to compute average annual impact"
            )
        if self.value_unit not in ["CHF", "EUR", "USD"]:
            raise ValueError(
                "To compute average annual impact, unit must be a currency."
            )

        # choose time interval such that two events coocurring are negligible
        time_window = 0.0001
        # frequencies = frequency_from_exceedance_frequency(self.exceedance_frequencies)
        probabilities = prob_from_exceedance_frequency(
            self.exceedance_frequencies, coincidence_fraction=time_window
        )
        probabilities = np.delete(probabilities, 0, axis=-1)
        return np.nansum(probabilities * self.values) / time_window

    def plot_return_period_curve(self):
        fig, ax = plt.subplots()
        ax.plot(self.values, 1 / self.exceedance_frequencies)
        ax.set_yscale("log")
        ax.set_xlabel(f"Exceedance value ({self.value_unit})")
        ax.set_ylabel(f"Return Period ({self.time_unit})")
        return fig, ax

    def plot_exceedance_instensity_curve(self):
        fig, ax = plt.subplots()
        ax.plot(1 / self.exceedance_frequencies, self.values)
        ax.set_xscale("log")
        ax.set_xlabel(f"Return Period ({self.time_unit})")
        ax.set_ylabel(f"Exceedance value ({self.value_unit})")
        return fig, ax


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
    # prepare values
    values = np.array(
        [return_period_curve.values for return_period_curve in exceedance_curves]
    )
    if value_resolution is None:
        value_resolution = np.nanmin(np.diff(values, axis=1))
    # round values to resolution
    value_bins = np.arange(
        np.nanmin(values), np.nanmax(values) + value_resolution, value_resolution
    )
    values = round_to_array(values, value_bins)
    # add zeros corresponding to nothing happens probability
    values = np.insert(values, 0, 0.0, axis=-1)

    # preapre probabilities
    exceedance_frequencies = np.array(
        [
            exceedance_curves.exceedance_frequencies
            for exceedance_curves in exceedance_curves
        ]
    )
    # convert to probabilities
    probabilities = prob_from_exceedance_frequency(
        exceedance_frequency=exceedance_frequencies,
        coincidence_fraction=coincidence_fraction,
    )

    # compute aggreagted values and probabilties
    final_values = values[0]
    final_probabilities = probabilities[0]
    for j in range(1, len(values)):
        final_values, final_probabilities = _combine_two_prob_sets(
            [values[j], probabilities[j]],
            [final_values, final_probabilities],
            aggregation_method,
            value_resolution,
        )

    final_exceedance_frequency = exceedance_frequency_from_prob(
        final_probabilities, coincidence_fraction=coincidence_fraction
    )
    # remove nothing happens bin
    final_values = final_values[1:]

    aggregated_return_period_curve = ExceedanceCurve(
        values=final_values,
        exceedance_frequencies=final_exceedance_frequency,
        time_unit=exceedance_curves[0].time_unit,
        value_unit=exceedance_curves[0].value_unit,
    )

    return aggregated_return_period_curve


def _combine_two_prob_sets(
    probabilistic_set1,
    probabilistic_set2,
    aggregation_method,
    value_resolution,
):
    """

    Parameters
    ----------
    values : _type_
        _description_
    probabilities : _type_
        _description_
    aggregation_method : _type_
        _description_
    """
    values = [probabilistic_set1[0], probabilistic_set2[0]]
    probabilities = [probabilistic_set1[1], probabilistic_set2[1]]

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
    # sum up corresponding probabilities
    corresponding_probabilities = np.array(
        [
            aggregated_probabilities[np.where(indices == index)].sum()
            for index in range(indices.max() + 1)
        ]
    )
    # reshape  probabilities
    final_probabilities = np.zeros_like(aggregated_bins)
    for unique_value, corresponding_probabilitiy in zip(
        unique_values, corresponding_probabilities
    ):
        final_probabilities[np.where(aggregated_bins == unique_value)] = (
            corresponding_probabilitiy
        )

    return (aggregated_bins, final_probabilities)
