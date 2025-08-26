"""
ExceedanceCurve class and corresponding functions
"""

from math import prod
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

import exceedance_curve_tools.utils as utils


class ExceedanceCurve:
    """ExceedanceCurve class"""

    def __init__(self, values, exceedance_frequencies, time_unit=None, value_unit=None):
        """Initialize Exceedance Curve instance.

        Parameters
        ----------
        values : np.ndarray of float
            Values corresponding to intensity or imapct
        exceedance_frequencies : np.ndarray of float
            exceedance frequencies corresponding to values
        time_unit : str, optional
            Time unit of the exceedance frequencies. Defaults to "year".
        value_unit : str, optional
            Values unit of the values. Defaults to "USD".
        """

        if len(values) != len(exceedance_frequencies):
            raise ValueError(
                f"Number of threshold values {len(values)} different to"
                " number of exceedance frequencies {len(exceedance_frequencies)}"
            )
        self.values = values.astype(float)
        self.exceedance_frequencies = exceedance_frequencies
        self.time_unit = time_unit if time_unit is not None else "year"
        self.value_unit = value_unit if value_unit is not None else "USD"

    def average_annual_impact(self, coincidence_fraction=None):
        """Compute average annual impact from exceedance impact curve

        Parameters
        ----------
        coincidence_fraction : float, optional
            Time window (as a fraction of the time unit) from which to compute probabilities.
            During this time window, the occurence of several impact will be neglected, and
            only the largest is considered. By default, exceedance frequencies will be converted
            to frequencies, which will be multiplied and summed for the AAI. This corresponds to
            choosing a very small coincidence_fraction.

        Returns
        -------
        float
            average annual imapct
        """
        if self.time_unit != "year":
            raise ValueError(
                "Time unit must year 'year' to compute average annual impact"
            )
        if self.value_unit not in ["CHF", "EUR", "USD"]:
            raise ValueError(
                f"Value unit {self.value_unit} not recognized (must be of CHF, EUR, or USD). "
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

    def plot_return_period_curve(self, axis=None):
        """Plot return period  curve (return period over impact or intensity)

        Parameters
        ----------
        axis : Axes, optional
            by default None

        Returns
        -------
        fig, axis
        """
        if axis is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = axis.get_figure(), axis

        ax.plot(self.values, 1 / self.exceedance_frequencies)
        ax.set_yscale("log")
        ax.set_xlabel(f"Exceedance value ({self.value_unit})")
        ax.set_ylabel(f"Return Period ({self.time_unit})")
        return fig, ax

    def plot_exceedance_values_curve(self, axis=None):
        """Plot exceedance curve (impact or intensity over return period)

        Parameters
        ----------
        axis : Axes, optional
            by default None

        Returns
        -------
        fig, axis
        """
        if axis is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = axis.get_figure(), axis

        ax.plot(1 / self.exceedance_frequencies, self.values)
        ax.set_xscale("log")
        ax.set_xlabel(f"Return Period ({self.time_unit})")
        ax.set_ylabel(f"Exceedance value ({self.value_unit})")
        return fig, ax


def combine_exceedance_curves(
    exceedance_curves,
    aggregation_method=sum,
    coincidence_fraction=1 / 12,
    use_sampling=True,
    correlation_factor=0.0,
    n_samples=10000,
    value_resolution=None,
):
    """Method to combine a number of exceedance curves

    Parameters
    ----------
    exceedance_curves : iterable of ExceedanceCurve
        The exceedance curves to be combined
    aggregation_method : callable, optional
        Way to combine different values if they happend at the same time.
        Defaults to sum.
    coincidence_fraction : float, optional
        Time window (as a fraction of the time unit) for which to consider two events as
        coincident and to combine their values. During this time window, the occurence of
        several events from each exceedance curve will be neglected, and the largest value
        for each is considered.
        Defaults to 1/12.
    use_sampling : bool, optional
        If True, combined exceedance curve will be sampled. If False, the exceedance curves will be
        combined by multiplying all corresponding probabilities. Defaults to True.
    correlation_factor : float, optional
        Only applied if use_sampling=True. In the sampling of from the different exceedance curves,
        a correlation factor is applied. If 1., sampled values are perfecly correlated (e.g., the
        largest values are drawn together). If 0., sampling is independent. If -1., sampled values
        are anticorrelated. Defaults to 0.
    n_samples : int, optional
        Only applied if use_sampling=True. Number of samples to use for the estimation
        of the combined exceedance curve. Defaults to 10000.
    value_resolution : float, optional
        Only applied if use_sampling=False. Resultion of the values to use when computing and
         aggregating all different combinations of values. Defaults to None.

    Returns
    -------
    ExceedanceCurve
        combined exceedance curve
    """
    # prepare values
    values = np.array(
        [return_period_curve.values for return_period_curve in exceedance_curves]
    )
    # fill NaN edges
    if np.any(np.isnan(values)):
        values = np.array([utils.fill_edges(row) for row in values])

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
        final_values, exceedance_probabilities = (
            _exceedance_probabilities_agg_from_sample(
                sampled_values, aggregation_method
            )
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
    """Combining two probabilistic sets by mutliplication of all possible combinations and
    aggregation the values, using a given value resolution"""
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
    """Sampling n_samples samples from different probabilitic sets (each including
    values and corresponding probabilities), using a correlation factor."""
    vals = np.flip(values, axis=-1)
    ex_freq = np.flip(exceedance_probabilities, axis=-1)
    n_prob_sets = vals.shape[0]
    quantile_samples = utils.get_correlated_quantiles(
        n_prob_sets, correlation_factor, n_samples
    ).T

    # Use searchsorted to find how many quantiles each sample surpasses
    indices = np.array(
        [
            np.searchsorted(ex_freq[j], quantile_samples[j], side="left")
            for j in range(n_prob_sets)
        ]
    )

    sampled_values = np.array([vals[j][index] for j, index in enumerate(indices)])
    return sampled_values


def _exceedance_probabilities_agg_from_sample(
    sampled_values,
    aggregation_method,
):
    """compute exceedance probabilities from the aggragation of values"""
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
