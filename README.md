# Setting Expectations for Hydrologic Model Performance with an Ensemble of Simple Benchmarks
This repository contains the analysis code for the paper:
```
Knoben, W. J. M. (accepted). Setting Expectations for Hydrologic Model Performance
with an Ensemble of Simple Benchmarks. Hydrologic Processes. DOI: 10.1002/hyp.15288
```

## Computational environment
The analysis uses both Python (for the benchmarks and plotting), and Matlab (for the model calibration part).

### Python
Note: analysis was performed while the `hydrobm` package was still under development and not available on `pypi` yet, so the `hydrobm` imports in the Python code are based on a situation where the `hydrobm` code is on the system but not installed by a package manager. Adjust accordingly if needed.

Python packages used:
```
cartopy geopandas matplotlib multiprocessing numpy os pandas seaborn time pathlib xarray
```

### Matlab
Script 6 was developed on Matlab, but with minor changes this could be run under the open-source Matlab alternative Octave. See the MARRMoT documentation for more details.

## Code explanation
This briefly describes the purpose of the various scripts in this repository. See below for code changes needed to re-run this analysis.

- `1_create_caravan_benchmarks.py`: Runs the `HydroBM` package for the Caravan v1.4 basins to create benchmark models for each basin, and calculate their associated NSE, KGE, MSE and RMSE scores.
- `2_create_score_summary_df.py`: Creates a summary of the scores calculated by script 1, to be used for further processing.
- `3_exploratory_analysis.py`: Some exploratory analysis, kept for posterity.
- `4_bm_example_plot.ipynb`: Creates Figure 1, an example of various benchmarks. Supposed to be run from the main Caravan directory, where script 1 created a new `benchmarks` folder at the same level as the existing CARAVAN `timeseries` directory.
- `5_large_domain_plots.py`: Creates a number of figures to explore benchmark performance across space, including paper Figures 2 (lines 108-217), 3 (lines 220-289) and 5 (lines 394-453 - HydroATLAS data).
- `6_calibrate_marrmot_models.m`: Calibrates five MARRMoT models for the example basin used in script 4.
- `7_model_comparison_plot.ipynb`: Creates Figure 4, comparing models against benchmarks. Supposed to be run from the main Caravan directory, where script 6 created a new `simulations` folder at the same level as the existing `timeseries` and `benchmarks` directories.

## Data and code requirements
This commentary uses the CARAVAN data set v1.4 (Kratzert et al., 2023, 2024) and the latest version of the MARRMoT toolbox available on its [GitHub repository](https://github.com/wknoben/MARRMoT/) (Knoben et al., 2019; Trotter et al., 2022).

When attempting to replicate this work, download the CARAVAN data manually and update data locations in the code:
- https://github.com/wknoben/benchmark_paper/blob/878a7227d8608c0dc5dc549452fddcc38a83614c/1_create_caravan_benchmarks.py#L17
- https://github.com/wknoben/benchmark_paper/blob/878a7227d8608c0dc5dc549452fddcc38a83614c/2_create_score_summary_df.py#L9
- https://github.com/wknoben/benchmark_paper/blob/878a7227d8608c0dc5dc549452fddcc38a83614c/3_exploratory_analysis.py#L10
- https://github.com/wknoben/benchmark_paper/blob/878a7227d8608c0dc5dc549452fddcc38a83614c/5_large_domain_plots.py#L15
- https://github.com/wknoben/benchmark_paper/blob/878a7227d8608c0dc5dc549452fddcc38a83614c/5_large_domain_plots.py#L29
- https://github.com/wknoben/benchmark_paper/blob/878a7227d8608c0dc5dc549452fddcc38a83614c/6_calibrate_marrmot_models.m#L9
    - **Note**: The path used in this script is different from the CARAVAN paths in the other scripts as a consequence of running on two different computational environments. Changing this to your CARAVAN directory (same path you'll use in the other scripts) should be correct.
 
Similarly, clone the MARRMoT repository and update the relevant paths in the code:
- https://github.com/wknoben/benchmark_paper/blob/878a7227d8608c0dc5dc549452fddcc38a83614c/6_calibrate_marrmot_models.m#L3


Further path changes needed:
- If script 1 was run on HPC, this path is where the log files are expected to be. These contain the runtime. Disable this part of the code if no log files are available.
    - https://github.com/wknoben/benchmark_paper/blob/878a7227d8608c0dc5dc549452fddcc38a83614c/3_exploratory_analysis.py#L24

## References
Knoben, W. J. M., Freer, J. E., Fowler, K. J. A., Peel, M. C., and Woods, R. A. (2019). Modular Assessment of Rainfall–Runoff Models Toolbox (MARRMoT) v1.2: an open-source, extendable framework providing implementations of 46 conceptual hydrologic models as continuous state-space formulations, Geosci. Model Dev., 12, 2463–2480, https://doi.org/10.5194/gmd-12-2463-2019

Kratzert, F., Nearing, G., Addor, N., Erickson, T., Gauch, M., Gilon, O., Gudmundsson, L., Hassidim, A., Klotz, D., Nevo, S., Shalev, G., & Matias, Y. (2023). Caravan—A global community dataset for large-sample hydrology. Scientific Data, 10(1), 61. https://doi.org/10.1038/s41597-023-01975-w

Kratzert, F., Nearing, G., Addor, N., Erickson, T., Gauch, M., Gilon, O., Gudmundsson, L., Hassidim, A., Klotz, D., Nevo, S., Shalev, G., & Matias, Y. (2024). Caravan—A global community dataset for large-sample hydrology (Version 1.4) [Dataset]. Zenodo. https://doi.org/10.5281/ZENODO.6522634

Trotter, L., Knoben, W. J. M., Fowler, K. J. A., Saft, M., and Peel, M. C. (2022). Modular Assessment of Rainfall–Runoff Models Toolbox (MARRMoT) v2.1: an object-oriented implementation of 47 established hydrological models for improved speed and readability, Geosci. Model Dev., 15, 6359–6369, https://doi.org/10.5194/gmd-15-6359-2022

## Funding
This project received funding under award NA22NWS4320003 from the NOAA Cooperative Institute Program. The statements, findings, conclusions, and recommendations are those of the author(s) and do not necessarily reflect the views of NOAA.
