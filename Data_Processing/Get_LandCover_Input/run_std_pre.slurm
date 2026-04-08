#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=200GB
#SBATCH --job-name="add_buffer"
#SBATCH --output=job_output/job-%j-output.txt

#SBATCH --container-mounts=/storage1/fs1/rvmartin2/Active/yany1:/my-projects2
#SBATCH --container-image=1yuyan/pytorch:latest

#SBATCH --container-workdir=/my-projects2/1.project/NO2_DL_US_2019/NO2_US_pkg/Data_Processing/Get_LandCover_Input


bash -lc "python3 aggreagte_LandType_variables.py"