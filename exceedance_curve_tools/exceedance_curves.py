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

    def __init__(
        self,
        values,
        exceedance_frequencies,
        time_unit=None,
        value_unit=None,
        aggregation_time_fraction=None,
    ):
        """Initialize Exceedance Curve instance.

        Parameters
        ----------
        values : np.ndarray of float
            Values corresponding to intensity or impact
        exceedance_frequencies : np.ndarray of float
            Exceedance frequencies corresponding to values
        time_unit : str, optional
            Time unit of the exceedance frequencies. Defaults to "year".
        value_unit : str, optional
            Values unit of the values. Defaults to "USD".
        aggregation_time_fraction : float or None, optional
            Fraction of the time_unit attribued used to aggregate values, e.g.
            1/12 for monthly aggregation when time_unit="year".
            None for single events. Defaults to None.
        """

        if len(values) != len(exceedance_frequencies):
            raise ValueError(
                f"Number of threshold values {len(values)} different to "
                f"number of exceedance frequencies {len(exceedance_frequencies)}"
            )
        self.values = values.astype(float)
        self.exceedance_frequencies = exceedance_frequencies
        self.time_unit = time_unit if time_unit is not None else "year"
        self.value_unit = value_unit if value_unit is not None else "USD"
        self.aggregation_time_fraction = aggregation_time_fraction

    def average_annual_impact(self, coincidence_fraction=None):
        """Compute average annual impact from exceedance impact curve

        Parameters
        ----------
        coincidence_fraction : float, optional
            Only effective if aggregation_time_fraction of ExceedanceCurve is not None.
            Time window (as a fraction of the time unit) from which to compute probabilities.
            During this time window, the occurrence of several impacts will be neglected, and
            only the largest is considered. By default, exceedance frequencies will be converted
            to frequencies, which will be multiplied and summed for the AAI. This corresponds to
            choosing a very small coincidence_fraction.

        Returns
        -------
        float
            Average annual impact
        """
        if self.value_unit not in ["CHF", "EUR", "USD"]:
            raise ValueError(
                f"Value unit {self.value_unit} not recognized (must be CHF, EUR, or USD). "
                "To compute average annual impact, unit must be a currency."
            )
        if (
            self.aggregation_time_fraction is not None
            and coincidence_fraction is not None
        ):
            agg_per = utils.aggregation_period(
                self.time_unit, self.aggregation_time_fraction
            )
            raise ValueError(
                f"The exceedance curve corresponds to total impact aggregated per {agg_per}. "
                "Use coincidence_fraction only if the exceedance curve corresponds to single events."
            )
        # Compute frequencies
        if coincidence_fraction:
            frequencies = (
                utils.prob_from_ex_freq(
                    self.exceedance_frequencies,
                    coincidence_fraction=coincidence_fraction,
                )[1:]
                / coincidence_fraction
            )
        else:
            frequencies = utils.freq_from_ex_freq(self.exceedance_frequencies)

        return np.nansum(frequencies * self.values)

    def plot_return_period_curve(self, axis=None, **kwargs):
        """Plot return period  curve (return period over impact or intensity)

        Parameters
        ----------
        axis : Axes, optional
            by default None
        kwargs : kwargs for plt.plot

        Returns
        -------
        fig, axis
        """
        if axis is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = axis.get_figure(), axis

        ax.plot(self.values, 1 / self.exceedance_frequencies, **kwargs)
        if self.aggregation_time_fraction is None:
            ax.set_xlabel(f"Exceedance value ({self.value_unit})")
        else:
            agg_per = utils.aggregation_period(
                self.time_unit, self.aggregation_time_fraction
            )
            ax.set_xlabel(
                f"Exceedance value of total impacts per {agg_per} ({self.value_unit})"
            )
        ax.set_ylabel(f"Return Period ({self.time_unit})")
        return fig, ax

    def plot_exceedance_curve(self, axis=None, **kwargs):
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

        ax.plot(1 / self.exceedance_frequencies, self.values, **kwargs)
        # ax.set_yscale("log")
        ax.set_xlabel(f"Return Period ({self.time_unit})")
        if self.aggregation_time_fraction is None:
            ax.set_ylabel(f"Exceedance value ({self.value_unit})")
        else:
            agg_per = utils.aggregation_period(
                self.time_unit, self.aggregation_time_fraction
            )
            ax.set_ylabel(
                f"Exceedance value of total impacts per {agg_per} ({self.value_unit})"
            )
        return fig, ax

    def sum_to_aggregation_time_fraction(
        self, aggregation_time_fraction, n_sampled_periods, rng=None
    ):
        """Sum single-event exceedance curve to total impact over aggregated periods.

        This method samples the total impact of random events occurring within an
        aggregation window defined by `aggregation_time_fraction` of the curve's
        `time_unit`, assuming that the number of events is Poisson distributed.
        The returned curve represents exceedance frequencies of total
        impacts over that aggregation window, and is tagged with the same
        `aggregation_time_fraction`.

        Parameters
        ----------
        aggregation_time_fraction : float
            Fraction of the curve's `time_unit` used for aggregation.
            For example, with `time_unit='year'`, use `1/12` for monthly aggregation.
        n_sampled_periods : int
            Number of aggregated periods to sample when estimating the aggregated
            exceedance curve.
        rng : numpy.random.Generator, optional
            Random number generator for sampling. If not provided, a new default
            generator is created.

        Returns
        -------
        ExceedanceCurve
            A new exceedance curve representing total impacts aggregated over the
            specified fraction of the time unit.
        """
        if self.aggregation_time_fraction is not None:
            agg_per = utils.aggregation_period(
                self.time_unit, self.aggregation_time_fraction
            )
            raise ValueError(
                "Only single-event exceedance curves can be aggregated. The current exceedance "
                f"curve is already aggreated to total impacts per {agg_per}."
            )
        if rng is None:
            rng = np.random.default_rng()

        frequencies = utils.freq_from_ex_freq(self.exceedance_frequencies)
        lambda_poisson = np.sum(frequencies) * aggregation_time_fraction
        n_events = rng.poisson(lam=lambda_poisson, size=n_sampled_periods)
        weights = frequencies / np.sum(frequencies)

        sampled_cum_impacts = np.array(
            [
                rng.choice(self.values, size=n, replace=True, p=weights).sum()
                for n in n_events
            ]
        )
        sampled_cum_impacts = np.sort(sampled_cum_impacts)
        ex_freq = (
            np.arange(1, n_sampled_periods + 1)[::-1]
            / n_sampled_periods
            / aggregation_time_fraction
        )

        return ExceedanceCurve(
            values=sampled_cum_impacts,
            exceedance_frequencies=ex_freq,
            time_unit=self.time_unit,
            value_unit=self.value_unit,
            aggregation_time_fraction=aggregation_time_fraction,
        )


def combine_exceedance_curves(
    exceedance_curves,
    aggregation_method=sum,
    coincidence_fraction=1 / 12,
    correlation_factor=0.0,
    n_samples=10000,
    n_sampled_periods=2000,
    rng=None,
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
    correlation_factor : float, optional
        Only applied if use_sampling=True. In the sampling of from the different exceedance curves,
        a correlation factor is applied. If 1., sampled values are perfecly correlated (e.g., the
        largest values are drawn together). If 0., sampling is independent. If -1., sampled values
        are anticorrelated. Defaults to 0.
    n_samples : int, optional
        Only applied if use_sampling=True. Number of samples to use for the estimation
        of the combined exceedance curve. Defaults to 10000.

    Returns
    -------
    ExceedanceCurve
        combined exceedance curve
    """
    if rng is None:
        rng = np.random.default_rng()

    # convert exceedance curves to aggregated curves if some are aggregated
    aggregation_time_fractions = np.array(
        [curve.aggregation_time_fraction for curve in exceedance_curves], dtype=float
    )

    curves_are_event_based = np.isnan(aggregation_time_fractions).all()
    if not curves_are_event_based:
        if (
            np.unique(
                aggregation_time_fractions[~np.isnan(aggregation_time_fractions)]
            ).size
            > 1
        ):
            raise ValueError(
                "Exceedance curves to combine correspond to different aggregation periods and cannot be combiend."
            )
        if coincidence_fraction is not None:
            raise ValueError(
                "Exceedance curves to combine correspond to aggregation period. Thus, coincidence_fraction cannot be used and must be set to None."
            )
        aggregation_time_fraction = np.unique(
            aggregation_time_fractions[~np.isnan(aggregation_time_fractions)]
        )[0]
        coincidence_fraction = 1
        exceedance_curves = np.array(
            [
                (
                    curve
                    if curve.aggregation_time_fraction is not None
                    else curve.sum_to_aggregation_time_fraction(
                        aggregation_time_fraction, n_sampled_periods, rng
                    )
                )
                for curve in exceedance_curves
            ]
        )

    # prepare values
    values = np.array(
        [return_period_curve.values for return_period_curve in exceedance_curves]
    )

    # prepare probabilities
    exceedance_frequencies = np.array(
        [
            exceedance_curves.exceedance_frequencies
            for exceedance_curves in exceedance_curves
        ]
    )

    # fill NaN edges
    if np.any(np.isnan(values)):
        values = np.array([utils.fill_edges(row) for row in values])

    # add zeros corresponding to nothing happens probability
    values = np.insert(values, 0, 0.0, axis=-1)

    # convert to probabilities
    exceedance_probabilities = utils.ex_prob_from_ex_freq(
        exceedance_frequencies,
        coincidence_fraction,
    )
    if not curves_are_event_based:
        exceedance_probabilities = exceedance_frequencies * aggregation_time_fraction

    # add exceedance_probability of nothing happening
    exceedance_probabilities = np.insert(exceedance_probabilities, 0, 1.0, axis=-1)
    sampled_values = _sample_from_prob_sets(
        values, exceedance_probabilities, correlation_factor, n_samples, rng
    )
    final_values, exceedance_probabilities = _exceedance_probabilities_agg_from_sample(
        sampled_values, aggregation_method
    )

    final_exceedance_frequency = utils.ex_freq_from_ex_prob(
        exceedance_probabilities, coincidence_fraction=coincidence_fraction
    )
    if not curves_are_event_based:
        final_exceedance_frequency = (
            exceedance_probabilities / aggregation_time_fraction
        )

    aggregated_return_period_curve = ExceedanceCurve(
        values=final_values,
        exceedance_frequencies=final_exceedance_frequency,
        time_unit=exceedance_curves[0].time_unit,
        value_unit=exceedance_curves[0].value_unit,
        aggregation_time_fraction=(
            None if curves_are_event_based else aggregation_time_fraction
        ),
    )

    return aggregated_return_period_curve


def combine_exceedance_curves_analytical(
    exceedance_curves,
    aggregation_method=sum,
    coincidence_fraction=1 / 12,
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
    value_resolution : float, optional
        Resultion of the values to use when computing and
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

    # convert to probabilities
    probabilities = utils.prob_from_ex_freq(
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

    final_exceedance_frequency = utils.ex_freq_from_prob(
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
    values, exceedance_probabilities, correlation_factor, n_samples, rng=None
):
    """Sampling n_samples samples from different probabilitic sets (each including
    values and corresponding probabilities), using a correlation factor."""
    if rng is None:
        rng = np.random.default_rng()
    vals = np.flip(values, axis=-1)
    ex_freq = np.flip(exceedance_probabilities, axis=-1)
    n_prob_sets = vals.shape[0]
    quantile_samples = utils.get_correlated_quantiles(
        n_prob_sets, correlation_factor, n_samples, rng
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
