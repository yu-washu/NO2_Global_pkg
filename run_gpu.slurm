#!/bin/bash
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --gpus=1
#SBATCH --mem=350GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yany1@wustl.edu
#SBATCH --job-name="LightGBM-wanb"
#SBATCH --output=job_output/wandb-LightGBM-%j-output.txt
#SBATCH --time=14-00:00:00
#SBATCH -vvv

#SBATCH --container-mounts=/storage1/fs1/rvmartin2/Active/yany1:/my-projects2
#SBATCH --container-image=1yuyan/pytorch-wandb:latest
#SBATCH --container-workdir=/my-projects2/1.project/NO2_DL_global/NO2_global_pkg

python3 main.py

# VERSION=v0.3.0

# # Redirect W&B and tmp paths inside the container
# export WANDB_DIR=/my-projects2/tmp/wandb_dir
# export TMPDIR=/my-projects2/tmp/tmpdir
# export XDG_CONFIG_HOME=/my-projects2/tmp/config_home
# export WANDB_CACHE_DIR=/my-projects2/tmp/wandb_cache
# export WANDB_AGENT_MAX_INITIAL_FAILURES=100
# mkdir -p $WANDB_DIR $TMPDIR $XDG_CONFIG_HOME $WANDB_CACHE_DIR

# # Create a frozen snapshot of current code
# TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# SNAPSHOT_BASE=/my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25/snapshots
# SNAPSHOT_DIR=$SNAPSHOT_BASE/$VERSION'_'$TIMESTAMP/

# echo "Creating snapshot at: $SNAPSHOT_DIR"
# mkdir -p $SNAPSHOT_DIR
# rsync -av --exclude=Results --exclude=job_output --exclude=Figures --exclude=wandb  /my-projects2/Projects/Daily_PM25_DL_2024/code/Training_Validation_Estimation/PM25/$VERSION/ $SNAPSHOT_DIR 

# # Run from the snapshot
# cd $SNAPSHOT_DIR
# echo "Running from snapshot: $SNAPSHOT_DIR"


# # Optional: mimic host exclusion if needed via constraints or scheduler filters

# # Run your code
# cd $SNAPSHOT_DIR

# # Random pause

# pause_time=$((RANDOM % 3 + 1))
# echo "Pausing for $pause_time seconds..."
# sleep $pause_time

# # Buffer_size=[80]
# # sed -i '/\[BLCO\-CrossValidation\]/,/^\[/{/Buffer_size/s/=.*/= '"$Buffer_size"'/}' config.toml
# nvidia-smi