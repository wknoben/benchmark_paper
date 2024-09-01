#!/bin/bash
#SBATCH --job-name=car_bm  				 # Job name
#SBATCH --output=create_caravan_benchmarks_out_run2.log  # Output file
#SBATCH --error=create_caravan_benchmarks_error_run2.log # Error file
#SBATCH --ntasks=1                      		 # Number of tasks (processes)
#SBATCH --cpus-per-task=32              		 # Number of CPU cores per task
#SBATCH --time=12:00:00                 		 # Maximum runtime (HH:MM:SS)
#SBATCH --account=hpc_c_giws_clark
#SBATCH --mail-user=wouter.knoben@usask.ca
#SBATCH --mail-type=ALL

# Load necessary modules
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 geo-stack/2023a

# Ensure HydroBM is accessible
export PYTHONPATH=$PYTHONPATH:/globalhome/wmk934/HPC/caravan_benchmarking/hydrobm

# Run the Python script
python /globalhome/wmk934/HPC/caravan_benchmarking/scripts/1_create_caravan_benchmarks.py
