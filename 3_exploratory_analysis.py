# Create plots

# --- Imports ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# --- Data location ---
main_dir = Path("/project/gwf/gwf_cmt/wknoben/caravan/Caravan")
run_name = "_run4_cal_end_at_half_timesteps_min_4year" # "_run1_cal_end_19990101", "_run2_cal_end_at_half_timesteps"
run_dir  = main_dir / "benchmarks" / run_name
run_file = run_dir / "benchmark_scores_summary.csv"
img_dir  = run_dir / "images"
img_dir.mkdir(exist_ok=True)

# --- Load the data ---
df = pd.read_csv(run_file, index_col=[0, 1])
metrics = df.index.get_level_values(0).unique()
benchmarks = df.columns

# --- How long did this run take? ---
# Anywhere between 80 and 215 seconds, with mean 184s.
log_file = Path("/globalhome/wmk934/HPC/caravan_benchmarking/logs/create_caravan_benchmarks_out_run4.log")
with open(log_file, 'r') as file:
    content = file.read()

times = np.fromstring(content[content.index('[') + 1: content.index(']')], sep=',')
no_data = (times == -1).sum() # check how often we aborted the benchmark run
times = times[times != -1] # remove the aborted runs
print(f'Number of no-data cases: {no_data}')
print(f'Minimum time: {times.min()}')
print(f'Mean time: {times.mean()}')
print(f'Std time: {times.std()}')
print(f'Maximum time: {times.max()}')

# --- Exploratory data analysis ---
# Per metric, plot boxplots of benchmark scores
for metric in metrics:
    df.loc[metric].boxplot(rot=90, figsize=(10, 6))
    plt.ylim(-3, 3)
    plt.title(f"{metric} scores")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(img_dir / f"boxplots_{metric}_scores.png")
    plt.close()

# Per metric, plot the best score in each basin as a line
for metric in metrics:
    sub_df = df.loc[metric].copy()
    best_scores = sub_df.max(axis=1)
    best_scores = best_scores.sort_values()
    best_scores = best_scores[~np.isnan(best_scores)]
    best_scores.plot(figsize=(10, 6))
    plt.title(f"{metric} best scores")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(img_dir / f"best_{metric}_scores.png")
    plt.close()

# Per metric, plot the relative ranks of each benchmark
for metric in metrics:
    sub_df = df.loc[metric].copy()
    if ("nse" in metric) or ("kge" in metric):
        ranks = sub_df.rank(axis=1, ascending=False) # high is better
    else:
        ranks = sub_df.rank(axis=1, ascending=True) # lower is better
    ranks.boxplot(rot=90, figsize=(10, 6))
    plt.title(f"{metric} ranks")
    plt.ylabel("Rank")
    plt.tight_layout()
    plt.savefig(img_dir / f"ranks_{metric}.png")
    plt.close()

# Per metric, create a 3x6 grid of scatter plots where each plot contains a "score vs rank" plot for a given benchmark
for metric in metrics:
    sub_df = df.loc[metric].copy()
    if ("nse" in metric) or ("kge" in metric):
        ranks = sub_df.rank(axis=1, ascending=False) # high is better
    else:
        ranks = sub_df.rank(axis=1, ascending=True) # lower is better
    fig, axs = plt.subplots(3, 6, figsize=(20, 10))
    for i, benchmark in enumerate(benchmarks):
        ax = axs[i // 6, i % 6]
        ax.scatter(sub_df[benchmark], ranks[benchmark], alpha=0.5, s=4)
        ax.set_title(benchmark)
        ax.set_xlabel("Score")
        ax.set_ylabel("Rank")
    plt.tight_layout()
    plt.savefig(img_dir / f"score_vs_rank_{metric}.png")
    plt.close()

# Per metric, plot the distribution of benchmark scores per basin
# SLOW
for metric in metrics:
    sub_df = df.loc[metric].copy()
    sub_df = sub_df.T # transpose to get basins as columns
    sub_df = sub_df.loc[sub_df.mean(axis=1).sort_values().index] # sort by mean score per basin
    sub_df.boxplot(rot=90, figsize=(10, 6))
    plt.title(f"{metric} scores per basin")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(img_dir / f"basin_scores_{metric}.png")
    plt.close()

# As above, but with lines for 5th, 25th, 50th, 75th, and 95th percentiles
for metric in metrics:
    sub_df = df.loc[metric].copy()
    sub_df = sub_df.T # transpose to get basins as columns
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    sub_df = sub_df.quantile(quantiles)
    sub_df = sub_df.sort_values(by=sub_df.index[2],axis=1) # sort by 50th percentile
    plt.figure(figsize=(10, 6))
    colors = ['darkgrey', 'lightgrey', 'black', 'lightgrey', 'darkgrey']
    for (index, row), q, c in zip(sub_df.iterrows(), quantiles, colors):
        plt.plot(row, label=f'{q:.2f}th percentile', color=c)
    if ("nse" in metric) or ("kge" in metric):
        plt.ylim(-1, 1)
    else:
        plt.ylim(0, 5)
    plt.legend(loc='upper right')
    plt.title(f"{metric} scores per basin")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(img_dir / f"basin_scores_percentiles_{metric}.png")
    plt.close()