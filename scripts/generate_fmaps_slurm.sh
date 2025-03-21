#!/bin/bash

#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --array=0-49
#SBATCH --mem=50G
#SBATCH --partition=intelsr_medium
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/DenoisingFunctionalMaps/SLURM_logs/generate_fmaps_%A_%a.out
#SBATCH --error=/home/s94zalek_hpc/DenoisingFunctionalMaps/SLURM_logs/generate_fmaps_%A_%a.err

# make the logs directory if it doesn't exist
mkdir -p /home/s94zalek_hpc/DenoisingFunctionalMaps/SLURM_logs

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/DenoisingFunctionalMaps
module load libGLU Xvfb
export PYTHONPATH="${PYTHONPATH}:/home/s94zalek_hpc/DenoisingFunctionalMaps"

###########################################
# basic parameters
###########################################

dataset_name=SMAL_sign_net_64_smal_old_data

sign_net_name=sign_net_64_smal_old_data

input_dir=/home/s94zalek_hpc/DenoisingFunctionalMaps/data/SMAL
output_dir=/lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train

template_type=animal

vis_freq=1000

###########################################
# Parallelization
###########################################

# calculate the start and end indices for the current job
dataset_size=64000

# Ensure at least one worker
worker_count=$((SLURM_ARRAY_TASK_COUNT > 1 ? SLURM_ARRAY_TASK_COUNT : 1))
curr_worker=$((SLURM_ARRAY_TASK_ID))

# Check if dataset_size is evenly divisible by worker_count
if (( dataset_size % worker_count != 0 )); then
    echo "Error: dataset_size ($dataset_size) is not evenly divisible by worker_count ($worker_count)" >&2
    exit 1
fi

# Compute start and end indices
chunk_size=$((dataset_size / worker_count))
idx_start=$((curr_worker * chunk_size))
idx_end=$((idx_start + chunk_size))

echo "Worker $curr_worker: Processing indices $idx_start to $idx_end, chunk size $chunk_size" >&2

###########################################
# Run the script
###########################################

# randomly sleep between 0 and 10 seconds to avoid overloading the file system
sleep_time=$((RANDOM % 10))
echo "Sleeping for ${sleep_time} seconds" >&2
sleep ${sleep_time}


# srun python denoisfm/data_generation/generate_fmaps.py --sign_net_name sign_net_64_norm_rm --input_dir /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/SURREAL --idx_start 0 --idx_end 100 --output_dir /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/SURREAL_sign_net_64_norm_rm --vis_freq 1 --template_type human
# rewrite line above but break it into multiple lines
# srun python denoisfm/data_generation/generate_fmaps.py \
#     --sign_net_name sign_net_64_norm_rm \
#     --input_dir /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/SURREAL \
#     --template_type human
#     --idx_start 0 \
#     --idx_end 100 \
#     --output_dir /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/SURREAL_sign_net_64_norm_rm \
#     --vis_freq 1 

srun python denoisfm/data_generation/generate_fmaps.py \
    --dataset_name $dataset_name \
    --sign_net_name $sign_net_name \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --template_type $template_type \
    --idx_start $idx_start \
    --idx_end $idx_end \
    --vis_freq $vis_freq
