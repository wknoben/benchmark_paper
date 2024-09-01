# Run the HydroBM benchmarks for the Caravan data

# --- Imports ---
import multiprocessing
import numpy as np
import os
import pandas as pd
import time
import xarray as xr

hydrobm_path = "/globalhome/wmk934/HPC/caravan_benchmarking/hydrobm"
os.chdir(hydrobm_path) # critical for the import to work
from hydrobm.calculate import calc_bm
from pathlib import Path

# --- Find data locations ---
main_dir = Path("/project/gwf/gwf_cmt/wknoben/caravan/Caravan")
data_dir = main_dir / "timeseries" / "netcdf"
data_folders = os.listdir(data_dir) # should be various subfolders
data_files = []
for folder in data_folders:
    data_files += [os.path.join(data_dir, folder, f) for f in os.listdir(os.path.join(data_dir, folder)) if f.endswith(".nc")]

# --- Create output locations ---
flow_dir = main_dir / "benchmarks" / "flows"
scor_dir = main_dir / "benchmarks" / "scores"
flow_dir.mkdir(exist_ok=True, parents=True)
scor_dir.mkdir(exist_ok=True, parents=True)

for folder in data_folders:
    (flow_dir / folder).mkdir(exist_ok=True, parents=True)
    (scor_dir / folder).mkdir(exist_ok=True, parents=True)

# --- Set the HydroBM specs ---
benchmarks_to_calculate = [
    # Streamflow benchmarks
    "mean_flow",
    "median_flow",
    "monthly_mean_flow",
    "monthly_median_flow",
    "daily_mean_flow",
    "daily_median_flow",
    # Long-term rainfall-runoff ratio benchmarks
    "rainfall_runoff_ratio_to_all",
    "rainfall_runoff_ratio_to_annual",
    "rainfall_runoff_ratio_to_monthly",
    "rainfall_runoff_ratio_to_daily",
    "rainfall_runoff_ratio_to_timestep",
    # Short-term rainfall-runoff ratio benchmarks
    "monthly_rainfall_runoff_ratio_to_monthly",
    "monthly_rainfall_runoff_ratio_to_daily",
    "monthly_rainfall_runoff_ratio_to_timestep",
    # Schaefli & Gupta (2007) benchmarks
    "scaled_precipitation_benchmark",  # equivalent to "rainfall_runoff_ratio_to_daily"
    "adjusted_precipitation_benchmark",
    "adjusted_smoothed_precipitation_benchmark",
]
metrics = ["nse", "kge", "mse", "rmse"]

# First run: fixed cal/val periods
# Downside: not all basins have data for the full period
# so we end up with some NaNs in the scores
#
#end_cal_date = np.datetime64("1999-01-01")

# --- Create the benchmark function ---
def run_benchmarks_for_file(data_file):

    # Start the timer
    start_time = time.time()

    # Load the data
    data = xr.open_dataset(data_file)

    # Second run: use half of the available data for calibration
    flow_mask = data['streamflow'] >= 0
    flow_dates = data['date'].values[flow_mask]
    median_date = flow_dates[len(flow_dates) // 2]

    # Ensure that the calibration and evaluation periods are at least 2 years each
    if len(flow_dates) < 4*365:
        return -1 # Exit, and return a clear error value 

    # Specify the calculation and evaluation periods, as boolean masks
    cal_mask = data["date"].values <= median_date
    val_mask = ~cal_mask      

    # Calculate the benchmarks and scores
    benchmarks, scores = calc_bm(
        data,
        # Time period selection
        cal_mask,
        val_mask=val_mask,
        # Variable names in 'data'
        precipitation="total_precipitation_sum",
        streamflow="streamflow",
        # Benchmarks and metrics
        benchmarks=benchmarks_to_calculate,
        metrics=metrics,
        # Snow model inputs
        calc_snowmelt=True,
        temperature="temperature_2m_mean",
        snowmelt_threshold=0.0,
        snowmelt_rate=3.0,
    )

    # Prep the dictionary for saving
    col_names = scores.pop("benchmarks", None)
    score_df = pd.DataFrame(scores, index=col_names)
    score_df = score_df.T
    score_df.index.name = "metric"

    # Save the benchmarks and scores
    basin_subfolder = os.path.dirname(data_file).split("/")[-1]
    file_name = os.path.basename(data_file).replace(".nc", "")
    benchmarks.to_csv(flow_dir / basin_subfolder / f"{file_name}.csv")
    score_df.to_csv(scor_dir / basin_subfolder / f"{file_name}.csv")
    
    # End the timer
    end_time = time.time()
    
    return end_time - start_time


# --- Run the benchmark in parallel ---
if __name__ == "__main__":
    # Create a pool of workers
    num_workers = 32
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Execute the function in parallel
        results = pool.map(run_benchmarks_for_file, data_files)

        print("Time taken:", results)
