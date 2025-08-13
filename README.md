**Summary**

*Exceedance frequency curves*. The central function method is `combine_exceedance_curves` in `exceedance_frequency_curves.py`.
This can be used to combine two exceedance impact or exceedance intensity curves with several parameters,
such as

- `coincidence_fraction`: The fraction of the time unit according to which we say that two events (impacts or intensities) occur at the same time
- `correlation_factor`: A correlation factor to control if the different distributions should be sampled independently or with some correlation.
- `aggregation_method`: How to combine the values of the different curves, e.g. `sum`for impact or `max` for intensities

These options are exemplified in `example_combine_two_curves.ipynb`.

Some thought on how to compute AAIs with exceedance impact curves are noted in `estimate_AAI.ipynb`.

*Return Period Maps*. One application of the above functionality is to change the resolution of return period maps in `return_period_maps.py`. E.g., if one wants to halven the
resultion, four exceedance curves must be combined into a single one, which can be done using the above functionality.

This is exemplified in `example_resolution.ipynb`.
