import matplotlib.pyplot as plt


class ExceedanceCurve:

    def __init__(self, values, exceedance_frequencies, time_unit=None, value_unit=None):

        if len(values) != len(exceedance_frequencies):
            raise ValueError(
                "Number of threshold values %s different to number of exceedance frequencies %s"
                % (len(values), len(exceedance_frequencies))
            )
        self.values = values
        self.exceedance_frequencies = exceedance_frequencies
        self.time_unit = time_unit if time_unit is not None else "year"
        self.value_unit = value_unit if value_unit is not None else "USD"

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
            )
            for i_centroid in range(local_exceedance_values.shape[0])
        ]
        return ReturnPeriodMap(exceedance_curves, geometry)
