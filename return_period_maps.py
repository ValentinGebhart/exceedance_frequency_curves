"""
ReturnPeriodMap class
"""

import numpy as np

from exceedance_curves import ExceedanceCurve, combine_exceedance_curves
from coordinates import change_grid_resolution


class ReturnPeriodMap:
    def __init__(self, exceedance_curves, geometry):
        self.exceedance_curves = exceedance_curves
        self.geometry = geometry

    def compute_aai_per_centroid(self, coincidence_fraction=None):
        return np.array(
            [
                curve.average_annual_impact(coincidence_fraction=coincidence_fraction)
                for curve in self.exceedance_curves
            ]
        )

    def compute_aai_aggregated(self, coincidence_fraction=None):
        return np.sum(
            self.compute_aai_per_centroid(coincidence_fraction=coincidence_fraction)
        )

    def coarsen(
        self,
        scale_factor,
        kwargs_combine_exceedance_curves=None,
    ):
        if kwargs_combine_exceedance_curves is None:
            kwargs_combine_exceedance_curves = {}
        new_geometry, assignment = change_grid_resolution(self.geometry, scale_factor)

        # default emtpy exceedance_curves TBD
        exceedance_curves = [
            combine_exceedance_curves(
                [
                    curve
                    for i, curve in enumerate(self.exceedance_curves)
                    if assignment[i] == j
                ],
                **kwargs_combine_exceedance_curves
            )
            for j in np.unique(assignment)
        ]

        new_geometry = new_geometry[
            [(j in assignment) for j in range(len(new_geometry))]
        ]

        return ReturnPeriodMap(exceedance_curves, new_geometry)

    @classmethod
    def from_CLIMADA_local_exceedance_intensity(cls, local_exceedance_intensity):
        """
        Create a ReturnPeriodMap from a CLIMADA local exceedance intensity object.

        Parameters:
        local_exceedance_intensity (DataFrame): DataFrame with columns as return periods and rows as centroids.

        Returns:
        ReturnPeriodMap: An instance of ReturnPeriodMap.
        """
        geometry = local_exceedance_intensity.geometry
        local_exceedance_values = local_exceedance_intensity.drop(columns="geometry")
        exceedance_curves = [
            ExceedanceCurve(
                values=local_exceedance_values.iloc[i_centroid, :].values,
                exceedance_frequencies=1
                / local_exceedance_values.columns.values.astype(float),
                # time_unit="years", # TBD: import this correctly
                # value_unit="USD", # TBD: import this correctly
            )
            for i_centroid in range(local_exceedance_values.shape[0])
        ]
        return ReturnPeriodMap(exceedance_curves, geometry)
