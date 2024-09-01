# Specific paper plots

# --- Imports ---
from cartopy import crs as ccrs, feature as cfeature
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from pathlib import Path

# --- Data location ---
main_dir = Path("/project/gwf/gwf_cmt/wknoben/caravan/Caravan")
run_name = "_run4_cal_end_at_half_timesteps_min_4year" # "_run1_cal_end_19990101", "_run2_cal_end_at_half_timesteps"
run_dir  = main_dir / "benchmarks" / run_name
run_file = run_dir / "benchmark_scores_summary.csv"
img_dir  = run_dir / "images" / "paper_plots"
img_dir.mkdir(exist_ok=True)

# --- Load the benchmark data ---
df = pd.read_csv(run_file, index_col=[0, 1])
df.replace([np.inf, -np.inf], np.nan, inplace=True) # prevents issues with histograms later
metrics = df.index.get_level_values(0).unique()
benchmarks = df.columns

# --- Get the basin attributes ---
att_dir = Path("/project/gwf/gwf_cmt/wknoben/caravan/Caravan/attributes")
sub_folders = os.listdir(att_dir)

merged = []
for sub_folder in sub_folders:
    att_car = pd.read_csv(att_dir / sub_folder / f'attributes_caravan_{sub_folder}.csv', index_col=0)
    att_hyd = pd.read_csv(att_dir / sub_folder / f'attributes_hydroatlas_{sub_folder}.csv', index_col=0)
    att_oth = pd.read_csv(att_dir / sub_folder / f'attributes_other_{sub_folder}.csv', index_col=0)
    merged.append(
        att_hyd.merge(
            att_oth.merge(att_car, left_index=True, right_index=True),
            left_index=True, right_index=True
            )
        )

attributes = pd.concat(merged)
attributes['aridity_ha'] = attributes['pet_mm_syr'] / attributes['pre_mm_syr']

# Prepare a plotting geodataframe, that has lat/lon and best score per basin
def prepare_best_score_plot_gdf(df,attributes,metric):
    # Get the data for the metric, and add location attributes
    sub_df = df.loc[metric].copy()
    best_scores = sub_df.max(axis=1)
    best_scores.name = f'best_{metric}'
    plot_df = attributes[['country','gauge_lat', 'gauge_lon']].merge(best_scores, left_index=True, right_index=True)
    plot_gdf = gpd.GeoDataFrame(plot_df[['country',best_scores.name]], geometry=gpd.points_from_xy(plot_df['gauge_lon'], plot_df['gauge_lat']))
    plot_gdf.crs = 'EPSG:4326' # set initial crs
    plot_gdf = plot_gdf[~np.isinf(plot_gdf[best_scores.name])] # Filter (-)inf values
    # Apply some metric-specific settings
    if ("nse" in metric) or ("kge" in metric):
        plot_gdf = plot_gdf.sort_values(by=best_scores.name) # highest on top
    else: # mse, rmse
        plot_gdf = plot_gdf.sort_values(by=best_scores.name, ascending=False) # lowest on top
    if "nse" in metric:
        plt_clim = (0, 1)
        hist_bins = np.linspace(-1, 1, 41)
        hist_xlim = (-0.1,1)
    elif "kge" in metric:
        plt_clim = (1-np.sqrt(2), 1)
        hist_bins = np.linspace(-1, 1, 41)
        hist_xlim = (1-np.sqrt(2), 1)
    else: # mse, rmse
        plt_clim = (0, 3)
        hist_bins = np.linspace(0, 3, 21)
        hist_xlim = plt_clim
    return plot_gdf, best_scores, plt_clim, hist_bins, hist_xlim

# # --- Plot 1: World maps of best scores ---
for metric in metrics:
    plot_gdf, best_scores, plt_clim, _, _ = prepare_best_score_plot_gdf(df,attributes,metric)
    plot_gdf = plot_gdf.to_crs(ccrs.Robinson())

    # Create the basemap in Robinson projection
    fig, ax = plt.subplots(1, 1, figsize=(14, 14), subplot_kw={'projection': ccrs.Robinson()})
    ax.coastlines()
    ax.stock_img()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='lightgrey')
    ax.set_global()

    # Plot the best score per basin
    ax.plot()
    plot_gdf.plot(ax=ax, column=best_scores.name, markersize=4, 
                legend=True, legend_kwds={
                        "location":"bottom",
                        "shrink":.6,
                        "orientation":"horizontal",},
                    cax=plt.gca().figure.add_axes([0.33, 0.28, 0.36, 0.01]),
                    vmin=plt_clim[0], vmax=plt_clim[1],)
                    #cax=plt.gca().figure.add_axes([0.3, 0.28, 0.4, 0.01]))

    ttl_metric = metric.split('_')[0].upper()
    ttl_period = "calculation" if metric.split('_')[1].lower() == 'cal' else "evaluation"
    ax.set_title(f"Best benchmark {ttl_period} {ttl_metric} scores per basin")

    # Save the map
    plt.savefig(img_dir / f"world_map_{metric}.png", bbox_inches='tight', dpi=300)
    plt.close()


# --- 1. Local maps of best scores, including histograms of cal/val NSE and KGE ---
# General settings
extents = [[-130, -70, 15, 76], # North America
           [-80, -35, -56, 1], # South America
           [-7, 4, 49, 60], # United Kingdom
           [7, 20, 46, 51], # Central Europe
           [115, 155, -45, -10]] # Australia
sq_extents = [[-130, -70, 15, 76],     # North America
              [-90, -30, -59, 10],     # South America
              [-12, 5, 49, 61],        # United Kingdom and surrounding area
              [8, 19, 43, 52],         # Central Europe and surrounding area
              [110, 160, -50, -5]]     # Australia and surrounding area
projections = [ccrs.LambertConformal(), # North America
            ccrs.LambertConformal(central_longitude=-60, standard_parallels=(-10, -30), cutoff=60), # South America
            ccrs.LambertConformal(central_longitude=0,   standard_parallels=(50, 60),   cutoff=-30), # United Kingdom
            ccrs.LambertConformal(central_longitude=15,  standard_parallels=(45, 55),   cutoff=-10), # Central Europe
            ccrs.LambertConformal(central_longitude=135, standard_parallels=(-18, -36), cutoff=60)] # Australia
titles = ["(a) CAMELS, HYSETS", "(b) CAMELS-CL, CAMELS-BR", "(c) CAMELS-GB", "(d) LamaH-CE", "(e) CAMELS-AUS"]

# Make the plots
for metric in metrics:

    # Get the metric name
    metric_name = metric.split('_')[0].upper()

    # Get the alternative metric for complete histograms
    if '_cal' in metric:
        hist1 = metric
        hist2 = metric.replace('_cal','_val')
    else:
        hist2 = metric
        hist1 = metric.replace('_val','_cal')

    # Get the data for this metric
    plot_gdf, best_scores, plt_clim, hist_bins, hist_xlim = prepare_best_score_plot_gdf(df,attributes,metric)

    # Open the figure
    fig = plt.figure(figsize=(8, 11))#, constrained_layout=True)
    gs = gridspec.GridSpec(3, 2, figure=fig)
    axes = []
    mini_axes = []

    # Create the local maps
    for (i,this_gs),extent,projection,ttl in zip(enumerate(gs), sq_extents, projections, titles):
        ax = fig.add_subplot(this_gs, projection=projection)
        #ax.stock_img() # terrain - doesn't work well with standard viridis colormap
        ax.add_feature(cfeature.OCEAN, color='lightskyblue')
        ax.add_feature(cfeature.LAND, color='lightgray')
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='k', alpha=0.5)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        tmp_gdf = plot_gdf.to_crs(projection)
        tmp_gdf.plot(ax=ax, column=best_scores.name, markersize=2, 
                    legend=True, legend_kwds={"shrink":.75,},
                        vmin=plt_clim[0], vmax=plt_clim[1],)
        ax.set_title(ttl)
        axes.append(ax)
        # Add small local histograms with cal and val comparisons
        cal_col = f'{metric_name.lower()}_cal'
        val_col = f'{metric_name.lower()}_val'
        cal_df = df.loc[cal_col].max(axis=1).copy()
        val_df = df.loc[val_col].max(axis=1).copy()
        if ('(a)' in ttl):
            local_mask = cal_df.index.get_level_values('basin').str.contains('camels_') | cal_df.index.get_level_values('basin').str.contains('hysets_')
        elif ('(b)' in ttl):
            local_mask = cal_df.index.get_level_values('basin').str.contains('camelscl_') | cal_df.index.get_level_values('basin').str.contains('camelsbr_')
        elif ('(c)' in ttl):
            local_mask = cal_df.index.get_level_values('basin').str.contains('camelsgb_')
        elif ('(d)' in ttl):
            local_mask = cal_df.index.get_level_values('basin').str.contains('lamah_')
        elif ('(e)' in ttl):
            local_mask = cal_df.index.get_level_values('basin').str.contains('camelsaus_')
        # Create new histogram axis at custom position
        nested_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=this_gs)
        mini_ax = fig.add_subplot(nested_gs[2, 0])
        hist_kge_cal,_,_ = mini_ax.hist(cal_df[local_mask], bins=hist_bins, color='lightblue', alpha=1.0, label='Calculation')
        hist_kge_val,_,_ = mini_ax.hist(val_df[local_mask], bins=hist_bins, color='orange', alpha=0.6, label='Evaluation')
        mini_ax.set_ylim(0, max(max(hist_kge_cal), max(hist_kge_val)) + 10)
        mini_ax.set_xlim(hist_xlim)
        mini_axes.append(mini_ax)

    # Add the cal and val histograms to the sixth subplot
    ax = fig.add_subplot(gs[2,1])
    hist_metric_cal,_,_ = ax.hist(df.loc[hist1].max(axis=1), bins=hist_bins, color='lightblue', alpha=1.0, label='Calculation')
    hist_metric_val,_,_ = ax.hist(df.loc[hist2].max(axis=1), bins=hist_bins, color='orange', alpha=0.65, label='Evaluation')
    ax.set_ylim(0, max(max(hist_metric_cal), max(hist_metric_val)) + 10)
    ax.set_xlim(hist_xlim)
    ax.set_title(f'(f) Benchmark {metric_name} scores')
    ax.set_xlabel(f'{metric_name}')
    ax.set_ylabel('All basins')
    ax.legend()
    axes.append(ax)

    # Move plots
    plt.tight_layout() # this sorts everything out at this point in plotting - next we move
    for ax,mini_ax in zip(axes[:5], mini_axes):
        # Move the mini axis
        ax_pos = ax.get_position()
        mini_pos = mini_ax.get_position()
        new_pos = [ax_pos.x0-0.025, ax_pos.y0-.025, mini_pos.width, mini_pos.height]
        mini_ax.set_position(new_pos) # put exactly in corner

    # Shrink the big histogram a little
    hist_pos = axes[5].get_position()
    new_pos = [hist_pos.x0, hist_pos.y0 + 0.1*hist_pos.height, hist_pos.width, 0.8*hist_pos.height]
    axes[5].set_position(new_pos)

    # Adjust the layout
    plt.savefig(img_dir / f"local_map_{metric}.png", bbox_inches='tight', dpi=300)
    plt.close()


## --- 2. Histograms of ranks, separated by data set ---
import warnings # because we're doing poorly with the xtick marks

# Per metric, plot the relative ranks of each benchmark
datasets = ["camels", "hysets", "camelscl", "camelsbr", "camelsgb", "lamah", "camelsaus"]
titles = ["(a) CAMELS", "(b) HYSETS", "(c) CAMELS-CL", "(d) CAMELS-BR", "(e) CAMELS-GB", "(f) LamaH-CE", "(g) CAMELS-AUS"]
x_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"]

colors = ['#7fc97f'] * 6 + ['#beaed4'] * 9 + ['#fdc086'] * 2
palette = dict(zip(df.columns, colors))
for metric in metrics:
    sub_df = df.loc[metric].copy()
    fig,axs = plt.subplots(7,1,figsize=(4,10))
    for i,(dataset,ttl) in enumerate(zip(datasets,titles)):
        data_mask = sub_df.index.str.contains(f"{dataset}_")
        dataset_df = sub_df[data_mask].copy()
        if ("nse" in metric) or ("kge" in metric):
            ranks = dataset_df.rank(axis=1, ascending=False) # high is better
        else:
            ranks = dataset_df.rank(axis=1, ascending=True) # lower is better
        ranks_melted = ranks.melt(var_name='benchmark',value_name='ranks')
        sns.boxplot(ax=axs[i], x='benchmark', y='ranks', data=ranks_melted, 
                    palette=palette, hue='benchmark', legend=False)
        axs[i].set_title(ttl)
        axs[i].set_ylabel("Rank")
        axs[i].set_xticklabels(x_labels)
        if i < 6: axs[i].set_xlabel("")
    plt.tight_layout()
    plt.savefig(img_dir / f"ranks_{metric}_per_dataset.png", bbox_inches='tight', dpi=300)
    plt.close()

# Double column version
datasets = ["camels", "hysets", "camelscl", "camelsbr", "camelsgb", "lamah", "camelsaus"]
titles = ["CAMELS", "HYSETS", "CAMELS-CL", "CAMELS-BR", "CAMELS-GB", "LamaH-CE", "CAMELS-AUS"]
x_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"]

colors = ['#7fc97f'] * 6 + ['#beaed4'] * 9 + ['#fdc086'] * 2
palette = dict(zip(df.columns, colors))
metric_pairs = [["kge_cal", "kge_val"], ["nse_cal", "nse_val"], ["rmse_cal", "rmse_val"], ["mse_cal", "mse_val"]]
subplot_ids = [['(a)', '(h)'], ['(b)', '(i)'], ['(c)', '(j)'], ['(d)', '(k)'], ['(e)', '(l)'], ['(f)', '(m)'], ['(g)', '(n)']]
for metric_pair in metric_pairs:
    fig,axs = plt.subplots(7,2,figsize=(8,10))
    for j in range(2):
        metric = metric_pair[j]
        sub_df = df.loc[metric].copy()
        for i,(dataset,ttl) in enumerate(zip(datasets,titles)):
            data_mask = sub_df.index.str.contains(f"{dataset}_")
            dataset_df = sub_df[data_mask].copy()
            if ("nse" in metric) or ("kge" in metric):
                ranks = dataset_df.rank(axis=1, ascending=False) # high is better
            else:
                ranks = dataset_df.rank(axis=1, ascending=True) # lower is better
            ranks_melted = ranks.melt(var_name='benchmark',value_name='ranks')
            sns.boxplot(ax=axs[i,j], x='benchmark', y='ranks', data=ranks_melted, 
                        palette=palette, hue='benchmark', legend=False)
            if i == 0: 
                ttl_metric = metric.split('_')[0].upper()
                ttl_period = "Calculation" if metric.split('_')[1].lower() == 'cal' else "Evaluation"
                ttl = f'{ttl_period} {ttl_metric} ranks \n{subplot_ids[i][j]} {ttl}'
            else:
                ttl = f'{subplot_ids[i][j]} {ttl}'
            axs[i,j].set_title(ttl)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                axs[i,j].set_xticklabels(x_labels)
            if j == 1: axs[i,j].set_ylabel("")
            if i < 6: axs[i,j].set_xlabel("")
    plt.tight_layout()
    plt.savefig(img_dir / f"ranks_{ttl_metric}_per_dataset_two_columns.png", bbox_inches='tight', dpi=300)
    plt.close()

# --- 3. Scatter plots of metric cal and val, against various attributes ---
metric_pairs = [["kge_cal", "kge_val"], ["nse_cal", "nse_val"], ["rmse_cal", "rmse_val"], ["mse_cal", "mse_val"]]

for metric_pair in metric_pairs:
    fig, axs = plt.subplots(3, 2, figsize=(8, 10))
    for i,metric in enumerate(metric_pair):
        # Prep the data
        sub_df = df.loc[metric].max(axis=1).copy()
        sub_df.name = metric
        sub_df = attributes.merge(sub_df, left_index=True, right_index=True)
        sub_df['dataset'] = sub_df.index.str.split('_').str[0]
        # Replace the 'dataset' column with better-looking names
        sub_df['dataset'] = sub_df['dataset'].replace({
            'camels': 'CAMELS',
            'hysets': 'HYSETS',
            'camelscl': 'CAMELS-CL',
            'camelsbr': 'CAMELS-BR',
            'camelsgb': 'CAMELS-GB',
            'lamah': 'LamaH-CE',
            'camelsaus': 'CAMELS-AUS'
        })
        # Plot metric against the three climate indices
        h1 = sns.scatterplot(data=sub_df, s=10, legend=True, x='moisture_index', y=metric, hue='dataset', ax=axs[0, i])
        h2 = sns.scatterplot(data=sub_df, s=10, legend=False, x='seasonality', y=metric, hue='dataset', ax=axs[1, i])
        h3 = sns.scatterplot(data=sub_df, s=10, legend=False, x='frac_snow', y=metric, hue='dataset', ax=axs[2, i])
        # Update the chart
        metric_name = metric.split('_')[0].upper()
        period = "Calculation" if metric.split('_')[1].lower() == 'cal' else "Evaluation"
        axs[0, i].set_title(f'{period} {metric_name}')
        axs[0, i].set_xlabel('Moisture index')
        axs[0, i].set_ylabel(metric_name)
        axs[1, i].set_xlabel('Seasonality')
        axs[1, i].set_ylabel(metric_name)
        axs[2, i].set_xlabel('Fraction snow')
        axs[2, i].set_ylabel(metric_name)
        h1.get_legend().set_title('')
        #h1.legend(loc='lower right', labels=['CAMELS', 'HYSETS', 'CAMELS-CL', 'CAMELS-BR', 'CAMELS-GB', 'LamaH-CE', 'CAMELS-AUS'])
    axs[0,0].get_legend().remove() # Keeping the top right one works well for KGE
    plt.tight_layout()
    plt.savefig(img_dir / f"3_index_climate_correlation_{metric_name.lower()}_two_columns.png", bbox_inches='tight', dpi=300)
    plt.close()

# --- 3. Scatter plots of metric cal and val, against aridity and fs ---
metric_pairs = [["kge_cal", "kge_val"], ["nse_cal", "nse_val"], ["rmse_cal", "rmse_val"], ["mse_cal", "mse_val"]]

for metric_pair in metric_pairs:
    fig, axs = plt.subplots(7, 4, figsize=(8, 11))
    for i,metric in enumerate(metric_pair):
        # Prep the data
        sub_df = df.loc[metric].max(axis=1).copy()
        sub_df.name = metric
        sub_df = attributes.merge(sub_df, left_index=True, right_index=True)
        sub_df['dataset'] = sub_df.index.str.split('_').str[0]
        # Replace the 'dataset' column with better-looking names
        sub_df['dataset'] = sub_df['dataset'].replace({
            'camels': 'CAMELS',
            'hysets': 'HYSETS',
            'camelscl': 'CAMELS-CL',
            'camelsbr': 'CAMELS-BR',
            'camelsgb': 'CAMELS-GB',
            'lamah': 'LamaH-CE',
            'camelsaus': 'CAMELS-AUS'
        })
        # Get the titles
        metric_name = metric.split('_')[0].upper()
        period = "Calculation" if metric.split('_')[1].lower() == 'cal' else "Evaluation"
        # Plot each dataset in its own subplot, against aridity and fraction snow
        for j,dataset in enumerate(['CAMELS', 'HYSETS', 'CAMELS-CL', 'CAMELS-BR', 'CAMELS-GB', 'LamaH-CE', 'CAMELS-AUS']):
            data_df = sub_df[sub_df['dataset'] == dataset]
            h1 = sns.scatterplot(data=data_df, s=10, legend=True, x='aridity', y=metric, ax=axs[j, i*2])
            h2 = sns.scatterplot(data=data_df, s=10, legend=False, x='frac_snow', y=metric, ax=axs[j, i*2+1])
            axs[j, i*2].axvline(x=1, color='orange') # line to indicate aridity = 1
            axs[j, i*2].set_xlim([-0.05,10.05])
            axs[j, i*2+1].set_xlim([-0.05,1.05])
            #if i == 0:
            #    axs[j, i*2].set_ylim([-0.05,1.05])
            #    axs[j, i*2+1].set_ylim([-0.05,1.05])
            #else:
            axs[j, i*2].set_ylim([-.55,1.05])
            axs[j, i*2+1].set_ylim([-.55,1.05])
            if j == 6:
                axs[j, i*2].set_xlabel('Aridity')
                axs[j, i*2+1].set_xlabel('Fraction snow')
            else:
                axs[j, i*2].set_xlabel('')
                axs[j, i*2+1].set_xlabel('')
            if i == 0:
                axs[j, i*2].set_ylabel(metric_name)
                axs[j, i*2+1].set_ylabel('')
            else:
                axs[j, i*2].set_ylabel('')
                axs[j, i*2+1].set_ylabel('')
            if j == 0: 
                axs[j, i*2].set_title(f'{period} {metric_name}\n{dataset}')
                axs[j, i*2+1].set_title(f'{period} {metric_name}\n{dataset}')
            else:
                axs[j, i*2].set_title(f'{dataset}')
                axs[j, i*2+1].set_title(f'{dataset}')
    plt.tight_layout()
    plt.savefig(img_dir / f"2_index__climate_correlation_{metric_name.lower()}_two_columns.png", bbox_inches='tight', dpi=300)
    plt.close()

# Same, but with HydroAtlas aridity P/PET
xlims = [[-0.05,5.05], # CAMELS
         [-0.05,5.05], # HYSETS
         [-0.05,10.05],# CAMELS-CL
         [-0.05,3.05], # CAMELS-BR
         [-0.05,1.55], # CAMELS-GB
         [-0.05,1.55], # LamaH-CE
         [-0.05,3.05]] # CAMELS-AUS ; determined after looking at plot initially

for metric_pair in metric_pairs:
    fig, axs = plt.subplots(7, 4, figsize=(8, 11))
    for i,metric in enumerate(metric_pair):
        # Prep the data
        sub_df = df.loc[metric].max(axis=1).copy()
        sub_df.name = metric
        sub_df = attributes.merge(sub_df, left_index=True, right_index=True)
        sub_df['dataset'] = sub_df.index.str.split('_').str[0]
        # Replace the 'dataset' column with better-looking names
        sub_df['dataset'] = sub_df['dataset'].replace({
            'camels': 'CAMELS',
            'hysets': 'HYSETS',
            'camelscl': 'CAMELS-CL',
            'camelsbr': 'CAMELS-BR',
            'camelsgb': 'CAMELS-GB',
            'lamah': 'LamaH-CE',
            'camelsaus': 'CAMELS-AUS'
        })
        # Get the titles
        metric_name = metric.split('_')[0].upper()
        period = "Calculation" if metric.split('_')[1].lower() == 'cal' else "Evaluation"
        # Plot each dataset in its own subplot, against aridity and fraction snow
        for j,dataset in enumerate(['CAMELS', 'HYSETS', 'CAMELS-CL', 'CAMELS-BR', 'CAMELS-GB', 'LamaH-CE', 'CAMELS-AUS']):
            data_df = sub_df[sub_df['dataset'] == dataset]
            h1 = sns.scatterplot(data=data_df, s=10, legend=True, x='aridity_ha', y=metric, ax=axs[j, i])
            h2 = sns.scatterplot(data=data_df, s=10, legend=False, x='frac_snow', y=metric, ax=axs[j, i+2])
            axs[j, i].axvline(x=1, color='orange') # line to indicate aridity = 1
            axs[j, i].set_xlim(xlims[j])
            axs[j,i+2].set_xlim([-0.05,1.05]) # fraction snow X
            axs[j, i].set_ylim([-.55,1.05]) # all KGEs Y
            axs[j, i+2].set_ylim([-.55,1.05])
            if j == 6:
                axs[j, i].set_xlabel('Aridity')
                axs[j, i+2].set_xlabel('Fraction snow')
            else:
                axs[j, i].set_xlabel('')
                axs[j, i+2].set_xlabel('')
            if i == 0:
                axs[j, i].set_ylabel(metric_name)
                axs[j, i+2].set_ylabel('')
            else:
                axs[j, i].set_ylabel('')
                axs[j, i+2].set_ylabel('')
            if j == 0: 
                axs[j, i].set_title(f'{period} {metric_name}\n{dataset}')
                axs[j, i+2].set_title(f'{period} {metric_name}\n{dataset}')
            else:
                axs[j, i].set_title(f'{dataset}')
                axs[j, i+2].set_title(f'{dataset}')
    plt.tight_layout()
    plt.savefig(img_dir / f"2_index_climate_correlation_{metric_name.lower()}_two_columns_hydroatlas_aridity.png", bbox_inches='tight', dpi=300)
    plt.close()


print(f"Completed plots in {img_dir}")

'''
BACKUP code for the local plots

# Open the figure
    fig = plt.figure(figsize=(8, 11), constrained_layout=True)
    gs = gridspec.GridSpec(3, 2, figure=fig)
    axes = []
    mini_axes = []

    # Create the local maps
    for (i,this_gs),extent,projection,ttl in zip(enumerate(gs), sq_extents, projections, titles):
        ax = fig.add_subplot(this_gs, projection=projection)
        #ax.stock_img() # terrain - doesn't work well with standard viridis colormap
        ax.add_feature(cfeature.OCEAN, color='lightskyblue')
        ax.add_feature(cfeature.LAND, color='lightgray')
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='k', alpha=0.5)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        tmp_gdf = plot_gdf.to_crs(projection)
        tmp_gdf.plot(ax=ax, column=best_scores.name, markersize=1, 
                    legend=True, legend_kwds={"shrink":.75,},
                        vmin=plt_clim[0], vmax=plt_clim[1],)
        ax.set_title(ttl)
        axes.append(ax)
        # Add small local histograms with cal and val comparisons
        cal_col = f'{metric_name.lower()}_cal'
        val_col = f'{metric_name.lower()}_val'
        cal_df = df.loc[cal_col].max(axis=1).copy()
        val_df = df.loc[val_col].max(axis=1).copy()
        if ('(a)' in ttl):
            local_mask = cal_df.index.get_level_values('basin').str.contains('camels_') | cal_df.index.get_level_values('basin').str.contains('hysets_')
        elif ('(b)' in ttl):
            local_mask = cal_df.index.get_level_values('basin').str.contains('camelscl_') | cal_df.index.get_level_values('basin').str.contains('camelsbr_')
        elif ('(c)' in ttl):
            local_mask = cal_df.index.get_level_values('basin').str.contains('camelsgb_')
        elif ('(d)' in ttl):
            local_mask = cal_df.index.get_level_values('basin').str.contains('lamah_')
        elif ('(e)' in ttl):
            local_mask = cal_df.index.get_level_values('basin').str.contains('camelsaus_')
        # Create new histogram axis at custom position
        #ax = fig.add_subplot(this_gs, position=[0.6, 0.6, 0.2, 0.2])
        nested_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=this_gs)
        mini_ax = fig.add_subplot(nested_gs[2, 0])
        hist_kge_cal,_,_ = mini_ax.hist(cal_df[local_mask], bins=hist_bins, color='lightblue', alpha=1.0, label='Calculation')
        hist_kge_val,_,_ = mini_ax.hist(val_df[local_mask], bins=hist_bins, color='orange', alpha=0.6, label='Evaluation')
        mini_ax.set_ylim(0, max(max(hist_kge_cal), max(hist_kge_val)) + 10)
        #pos = mini_ax.get_position()  # Get the current position
        #new_pos = [pos.x0 - .1, pos.y0 * 0.9, pos.width, pos.height]
        #mini_ax.set_position(new_pos) # move slightly out
        mini_axes.append(mini_ax)

    # Add the cal and val histograms to the sixth subplot
    ax = fig.add_subplot(gs[2,1])
    hist_kge_cal,_,_ = ax.hist(df.loc['kge_cal'].max(axis=1), bins=hist_bins, color='lightblue', alpha=1.0, label='Calculation')
    hist_kge_val,_,_ = ax.hist(df.loc['kge_val'].max(axis=1), bins=hist_bins, color='orange', alpha=0.65, label='Evaluation')
    ax.set_ylim(0, max(max(hist_kge_cal), max(hist_kge_val)) + 10)  
    ax.set_title(f'(f) Benchmark {metric_name} scores')
    ax.set_xlabel(f'{metric_name}')
    ax.set_ylabel('Basins')
    ax.legend()

'''