"""
ReturnPeriodMap class
"""

import numpy as np
import geopandas as gpd

import climada.util.interpolation as u_interp

from exceedance_curves import ExceedanceCurve, combine_exceedance_curves
from coordinates import change_grid_resolution
from utils import frequency_from_exceedance_frequency


class ReturnPeriodMap:
    """ReturnPeriodMapp class"""

    def __init__(self, exceedance_curves, geometry):
        """Initialize ReturnPeriodMap instance.

        Parameters
        ----------
        exceedance_curves : iterable of ExceedanceCurve
            exceedance curves for the different centroids
        geometry : GeometryArray
            geometry including the centroids
        """
        # check that value unit of exceedance curves is the same
        if not np.array_equal(
            [exceedance_curves[0].value_unit] * len(exceedance_curves),
            [ec.value_unit for ec in exceedance_curves],
        ):
            raise ValueError("Value unit of different exceedance curves are not equal.")
        # check that time unit of exceedance curves is the same
        if not np.array_equal(
            [exceedance_curves[0].time_unit] * len(exceedance_curves),
            [ec.time_unit for ec in exceedance_curves],
        ):
            raise ValueError("Time unit of different exceedance curves are not equal.")

        self.exceedance_curves = exceedance_curves
        self.geometry = geometry
        self.value_unit = exceedance_curves[0].value_unit
        self.time_unit = exceedance_curves[0].time_unit

    def compute_aai_per_centroid(self, coincidence_fraction=None):
        """Compute average annual impact per centroid

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
            average annual impact per centroid
        """
        return np.array(
            [
                curve.average_annual_impact(coincidence_fraction=coincidence_fraction)
                for curve in self.exceedance_curves
            ]
        )

    def compute_aai_aggregated(self, coincidence_fraction=None):
        """Compute average annual impact aggregated over all centroids

        Parameters
        ----------
        coincidence_fraction : float , optional
            Time window (as a fraction of the time unit) from which to compute probabilities.
            During this time window, the occurence of several impact will be neglected, and
            only the largest is considered. By default, exceedance frequencies will be converted
            to frequencies, which will be multiplied and summed for the AAI. This corresponds to
            choosing a very small coincidence_fraction.

        Returns
        -------
        float
            aggregated average annual imapct
        """
        return np.sum(
            self.compute_aai_per_centroid(coincidence_fraction=coincidence_fraction)
        )

    def coarsen(
        self,
        scale_factor,
        aggregation_method=sum,
        coincidence_fraction=1 / 12,
        use_sampling=True,
        correlation_factor=0.0,
        n_samples=10000,
        value_resolution=None,
    ):
        """Method to coarsen the resolution of a RerturnPeriodMap

        Parameters
        ----------
        scale_factor : float
            factor by which the resolution should be changed. E.g., scale_factor == 2 results
            in a grid with lon (lat) grid spacing equal to twice the minimal lon (lat)
            difference in intial geometry.
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
            If True, combined exceedance curve will be sampled. If False, the exceedance curves will
            be combined by multiplying all corresponding probabilities. Defaults to True.
        correlation_factor : float, optional
            Only applied if use_sampling=True. In the sampling of from the different exceedance
            curves, a correlation factor is applied. If 1., sampled values are perfecly correlated
            (e.g., the largest values are drawn together). If 0., sampling is independent. If -1.,
            sampled values are anticorrelated. Defaults to 0.
        n_samples : int, optional
            Only applied if use_sampling=True. Number of samples to use for the estimation
            of the combined exceedance curve. Defaults to 10000.
        value_resolution : float, optional
            Only applied if use_sampling=False. Resultion of the values to use when computing and
            aggregating all different combinations of values. Defaults to None.

        Returns
        -------
        RerturnPeriodMap
            coarsened RerturnPeriodMap instance
        """
        new_geometry, assignment = change_grid_resolution(self.geometry, scale_factor)

        # default emtpy exceedance_curves TBD
        exceedance_curves = [
            combine_exceedance_curves(
                [
                    curve
                    for i, curve in enumerate(self.exceedance_curves)
                    if assignment[i] == j
                ],
                aggregation_method=aggregation_method,
                coincidence_fraction=coincidence_fraction,
                use_sampling=use_sampling,
                correlation_factor=correlation_factor,
                n_samples=n_samples,
                value_resolution=value_resolution,
            )
            for j in np.unique(assignment)
        ]

        new_geometry = new_geometry[
            [(j in assignment) for j in range(len(new_geometry))]
        ]

        return ReturnPeriodMap(exceedance_curves, new_geometry)

    def get_local_exceedance_values(
        self,
        return_periods,
        label="Impact",
        method="extrapolate_constant",
        min_value=0,
        log_frequency=True,
        log_value=True,
        bin_decimals=None,
    ):
        """compute local exceedance values for each centroid for given return periods

        Parameters
        ----------
        return_periods : array_like
            User-specified return periods for which the exceedance intensity should be calculated
            locally (at each centroid).
        method : str
            Method to interpolate to new return periods. Currently available are "interpolate",
            "extrapolate", "extrapolate_constant" and "stepfunction". If set to "interpolate",
            return periods outside the range of the data's observed local return periods
            will be assigned NaN. If set to "extrapolate_constant" or "stepfunction",
            return periods larger than the data's local return periods will be
            assigned the largest local impact, and return periods smaller than the data's
            observed local return periods will be assigned 0. If set to "extrapolate", local
            exceedance values will be extrapolated (and interpolated). The extrapolation to
            large return periods uses the two highest values of the centroid and their return
            periods and extends the interpolation between these points to the given return period
            (similar for small return periods). Defauls to "extrapolate_constant".
        min_values : float, optional
            Minimum threshold to filter the values. Defaults to 0.
        log_frequency : bool, optional
            If set to True, (cummulative) frequency values are converted to log scale before
            inter- and extrapolation. Defaults to True.
        log_value : bool, optional
            If set to True, values are converted to log scale before
            inter- and extrapolation. Defaults to True.
        bin_decimals : int, optional
            Number of decimals to group and bin values. Binning results in smoother (and
            coarser) interpolation and more stable extrapolation. For more details and sensible
            values for bin_decimals, see Notes. If None, values are not binned. Defaults to None.


        Returns
        -------
        Geodataframe, str, callable
            Tuple consisting of local exceedane values, label of the dataframe, and a
            column-label-generating function (for reporting and plotting)
        """

        exceedance_frequency = 1 / np.array(return_periods)

        exceedance_values = np.array(
            [
                u_interp.preprocess_and_interpolate_ev(
                    exceedance_frequency,
                    None,
                    frequency_from_exceedance_frequency(
                        self.exceedance_curves[i_centroid].exceedance_frequencies
                    ),
                    self.exceedance_curves[i_centroid].values,
                    log_frequency=log_frequency,
                    log_values=log_value,
                    value_threshold=min_value,
                    method=method,
                    y_asymptotic=0.0,
                    bin_decimals=bin_decimals,
                )
                for i_centroid in range(len(self.exceedance_curves))
            ]
        )

        gdf = gpd.GeoDataFrame(geometry=self.geometry)
        col_names = [f"{ret_per}" for ret_per in return_periods]
        gdf[col_names] = exceedance_values

        # create label and column_label
        def column_label(column_names):
            return [f"Return Period: {col} {self.time_unit}" for col in column_names]

        return gdf, label + f" ({self.value_unit})", column_label

    @classmethod
    def from_climada_local_exceedance_impact(
        cls, local_exceedance_impact, time_unit=None, value_unit=None
    ):
        """
        Create a ReturnPeriodMap from a CLIMADA local exceedance impact gdf.

        Parameters
        ----------
        local_exceedance_impact : GeoDataFrame (DataFrame)
            GeoDataFrame with columns as return periods and rows as centroids. Each entry
            is the exceedance impact for a given centroid and return period.
        time_unit : str, optional
            time unit of the return periods. Defaults to "year".
        value_unit: str, optional
            value unit of the impacts. Defaults to "USD".

        Returns
        -------
        ReturnPeriodMap
            An instance of ReturnPeriodMap.
        """
        geometry = local_exceedance_impact.geometry
        local_exceedance_values = local_exceedance_impact.drop(columns="geometry")
        exceedance_curves = [
            ExceedanceCurve(
                values=local_exceedance_values.iloc[i_centroid, :].values,
                exceedance_frequencies=1
                / local_exceedance_values.columns.values.astype(float),
                time_unit=time_unit if time_unit is not None else "year",
                value_unit=value_unit if value_unit is not None else "USD",
            )
            for i_centroid in range(local_exceedance_values.shape[0])
        ]
        return ReturnPeriodMap(exceedance_curves, geometry)

    @classmethod
    def from_climada_local_exceedance_intensity(
        cls, local_exceedance_intensity, time_unit=None, value_unit=None
    ):
        """
        Create a ReturnPeriodMap from a CLIMADA local exceedance intensity gdf.

        Parameters
        ----------
        local_exceedance_intensity : GeoDataFrame (DataFrame)
            GeoDataFrame with columns as return periods and rows as centroids. Each entry
            is the exceedance intensity for a given centroid and return period.
        time_unit : str, optional
            time unit of the return periods. Defaults to "year".
        value_unit: str, optional
            value unit of the intensities. Defaults to "unit".

        Returns
        -------
        ReturnPeriodMap
            An instance of ReturnPeriodMap.
        """
        geometry = local_exceedance_intensity.geometry
        local_exceedance_values = local_exceedance_intensity.drop(columns="geometry")
        exceedance_curves = [
            ExceedanceCurve(
                values=local_exceedance_values.iloc[i_centroid, :].values,
                exceedance_frequencies=1
                / local_exceedance_values.columns.values.astype(float),
                time_unit=time_unit if time_unit is not None else "year",
                value_unit=value_unit if value_unit is not None else "unit",
            )
            for i_centroid in range(local_exceedance_values.shape[0])
        ]
        return ReturnPeriodMap(exceedance_curves, geometry)
