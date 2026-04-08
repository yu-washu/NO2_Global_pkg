#!/bin/bash
#SBATCH --partition=m7i-96-1b
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yany1@wustl.edu
#SBATCH --job-name="step2-Mahalanobis-plot"
#SBATCH --output=logs_v2-Uncertainty/step2-Mahalanobis-plot-%j.txt
#SBATCH --time=14-00:00:00
#SBATCH -vvv

mkdir -p logs_v2-Uncertainty
cd /fsx/yany1/1.project/NO2_DL_global/NO2_global_pkg/Mahalanobis_Uncertainty
source ~/miniconda3/etc/profile.d/conda.sh
conda activate python

ulimit -s unlimited

# python3 -u derive_resampled_trainingdata_BLISCO_data.py
# python3 -u calculate_the_mahalanobis_distance_binned_rRMSE.py
python3 -u derive_map_mahalanobis_uncertainty.py