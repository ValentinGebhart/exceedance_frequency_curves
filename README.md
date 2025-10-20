**Summary**

*Exceedance frequency curves*. The central function method is `combine_exceedance_curves` in `exceedance_frequency_curves.py`.
This function can be used to combine two exceedance impact or exceedance intensity curves with several parameters,
such as

- `coincidence_fraction`: The fraction of the time unit according to which we say that two events (impacts or intensities) occur at the same time
- `correlation_factor`: A correlation factor to control if the different distributions should be sampled independently or with some correlation.
- `aggregation_method`: How to combine the values of the different curves, e.g. `sum`for impact or `max` for intensities

These options are exemplified in `example_combine_two_curves.ipynb`.

*Return Period Maps*. One application of the above functionality is to change the resolution of return period maps in `return_period_maps.py`. E.g., if one wants to halven the
resultion, four exceedance curves must be combined into a single one, which can be done using the above functionality. This is exemplified in `change_resolution_RPMap.ipynb`.

**Requirements**

All provided functionality and demnostrator notebooks can be run using a python environment with the [`climada`](https://github.com/CLIMADA-project/climada_python) package installed. If CLIMADA-related functionality is not required, a python environment with `geopandas`, `matplotlib`, `numpy`, `scipy`, and `shapely` is sufficient.

**Demonstator notebooks**

`example_combine_two_curves.ipynb`. Presenting different ways to combine two exceedance curves.

`change_resolution_RPMap.ipynb`. How to coarsen return period maps using the exceedance curve combination functionality.

`estimate_AAI.ipynb`. Some thoughts on how to compute average annual impacts with exceedance impact curves. 


