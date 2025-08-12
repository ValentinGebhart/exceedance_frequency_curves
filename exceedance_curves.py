"""
ExceedanceCurve class and corresponding functions
"""

from math import prod
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

import utils


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

    def average_annual_impact(self, coincidence_fraction=None):
        """compute average annual impact"""
        if self.time_unit != "year":
            raise ValueError(
                "Time unit must year 'year' to compute average annual impact"
            )
        if self.value_unit not in ["CHF", "EUR", "USD"]:
            raise ValueError(
                "To compute average annual impact, unit must be a currency."
            )
        if coincidence_fraction:
            frequencies = (
                utils.prob_from_exceedance_frequency(
                    self.exceedance_frequencies,
                    coincidence_fraction=coincidence_fraction,
                )[1:]
                / coincidence_fraction
            )
        else:
            frequencies = utils.frequency_from_exceedance_frequency(
                self.exceedance_frequencies
            )

        return np.nansum(frequencies * self.values)

    def plot_return_period_curve(self):
        """plot return period  curve (return period over impact or intensity)"""
        fig, ax = plt.subplots()
        ax.plot(self.values, 1 / self.exceedance_frequencies)
        ax.set_yscale("log")
        ax.set_xlabel(f"Exceedance value ({self.value_unit})")
        ax.set_ylabel(f"Return Period ({self.time_unit})")
        return fig, ax

    def plot_exceedance_instensity_curve(self):
        """plot exceedance curve (impact or intensity over return period)"""
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
    use_sampling=True,
    correlation_factor=0.0,
    n_samples=10000,
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
    if not use_sampling:
        if value_resolution is None:
            value_resolution = np.nanmin(np.diff(values, axis=1))
        # round values to resolution
        value_bins = np.arange(
            np.nanmin(values), np.nanmax(values) + value_resolution, value_resolution
        )
        values = utils.round_to_array(values, value_bins)
    # add zeros corresponding to nothing happens probability
    values = np.insert(values, 0, 0.0, axis=-1)

    # preapre probabilities
    exceedance_frequencies = np.array(
        [
            exceedance_curves.exceedance_frequencies
            for exceedance_curves in exceedance_curves
        ]
    )

    if use_sampling:
        # convert to probabilities
        exceedance_probabilities = (
            utils.exceedance_probability_from_exceedance_frequency(
                exceedance_frequencies,
                coincidence_fraction,
            )
        )
        # add exceedance_probability of nothing happening
        exceedance_probabilities = np.insert(exceedance_probabilities, 0, 1.0, axis=-1)
        sampled_values = _sample_from_prob_sets(
            values, exceedance_probabilities, correlation_factor, n_samples
        )
        final_values, exceedance_probabilities = _exceedance_frequency_agg_from_sample(
            sampled_values, aggregation_method
        )

        final_exceedance_frequency = (
            utils.exceedance_frequency_from_exceedance_probability(
                exceedance_probabilities, coincidence_fraction=coincidence_fraction
            )
        )
    else:
        # convert to probabilities
        probabilities = utils.prob_from_exceedance_frequency(
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

        final_exceedance_frequency = utils.exceedance_frequency_from_prob(
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
    aggregated_values = utils.round_to_array(aggregated_values, aggregated_bins)
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


def _sample_from_prob_sets(
    values, exceedance_probabilities, correlation_factor, n_samples
):
    vals = np.flip(values, axis=-1)
    ex_freq = np.flip(exceedance_probabilities, axis=-1)
    # print("values", values)
    # print("exceedance_probabilities", exceedance_probabilities)
    n_prob_sets = vals.shape[0]
    quantile_samples = utils.get_correlated_quantiles(
        n_prob_sets, correlation_factor, n_samples
    ).T
    # print("samples", quantile_samples)
    # Use searchsorted to find how many quantiles each sample surpasses
    indices = np.array(
        [
            np.searchsorted(ex_freq[j], quantile_samples[j], side="left")
            for j in range(n_prob_sets)
        ]
    )

    sampled_values = np.array([vals[j][index] for j, index in enumerate(indices)])
    # print("sampled values")
    # print(sampled_values)
    return sampled_values


def _exceedance_frequency_agg_from_sample(
    sampled_values,
    aggregation_method,
):
    sampled_aggregated_values = np.apply_along_axis(
        aggregation_method, axis=0, arr=sampled_values
    )

    unique_sampled_aggregated_values = np.unique(sampled_aggregated_values)

    n = len(sampled_aggregated_values)

    # exceedance count
    exceedance_probs = np.array(
        [
            np.sum(sampled_aggregated_values >= x) / n
            for x in unique_sampled_aggregated_values
        ]
    )
    # remove nothing events if sampled
    if unique_sampled_aggregated_values[0] == 0:
        unique_sampled_aggregated_values = unique_sampled_aggregated_values[1:]
        exceedance_probs = exceedance_probs[1:]

    # print("vals", unique_sampled_aggregated_values )
    # print("exceedance_probs", exceedance_probs)

    return unique_sampled_aggregated_values, exceedance_probs
