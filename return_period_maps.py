"""
ReturnPeriodMap class
"""

from exceedance_curves import ExceedanceCurve


class ReturnPeriodMap:
    def __init__(self, exceedance_curves, geometry):
        self.exceedance_curves = exceedance_curves
        self.geometry = geometry

    def compute_aai_per_centroid(self):
        return

    def compute_aai_aggregated(self):
        return

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
