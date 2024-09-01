# Summarize the individual scores into a single dataframe

# --- Imports ---
import os
import pandas as pd
from pathlib import Path

# --- Data location ---
main_dir = Path("/project/gwf/gwf_cmt/wknoben/caravan/Caravan")
run_name = "_run4_cal_end_at_half_timesteps_min_4year" # "_run1_cal_end_19990101", ""_run2_cal_end_at_half_timesteps"
run_dir  = main_dir / "benchmarks" / run_name
flow_dir = run_dir / "flows"
scor_dir = run_dir / "scores"

# --- Find all score files ---
data_dirs = os.listdir(scor_dir)
score_files = []
for folder in data_dirs:
    score_files += [os.path.join(scor_dir, folder, f) for f in os.listdir(os.path.join(scor_dir, folder)) if f.endswith(".csv")]

# --- Generate a summary dataframe ---
basins = []
dfs = []
for score_file in score_files:
    basins.append(Path(score_file).stem) # Extract the basin identifier
    dfs.append(pd.read_csv(score_file))

multi_index_dfs = []
for score_file in score_files:
    df = pd.read_csv(score_file) # load the data
    df['basin'] = Path(score_file).stem # add the basin identifier
    df.set_index(['metric', 'basin'], inplace=True) # set the multi-index
    multi_index_dfs.append(df)

# Concatenate the dataframes into a multiindex dataframe, 
# with 'metric' as first level and 'basin' as second
final_df = pd.concat(multi_index_dfs)
final_df = final_df.sort_index()

# To file
final_df.to_csv(run_dir / "benchmark_scores_summary.csv")