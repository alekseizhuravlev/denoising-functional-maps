#!/bin/bash

#SBATCH -n 1
#SBATCH -t 5-00:00:00
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --partition=mlgpu_long
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/DenoisingFunctionalMaps/SLURM_logs/test_ddpm_%A_%a.out
#SBATCH --error=/home/s94zalek_hpc/DenoisingFunctionalMaps/SLURM_logs/test_ddpm_%A_%a.err

# make the logs directory if it doesn't exist
mkdir -p /home/s94zalek_hpc/DenoisingFunctionalMaps/SLURM_logs

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/DenoisingFunctionalMaps
module load libGLU Xvfb
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/DenoisingFunctionalMaps


exp_name=ddpm_64
checkpoint_name=epoch_99
data_dir=/lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/test

dataset_list=(
    'FAUST_r'
    'SCAPE_r'
    'SHREC19_r'
    'FAUST_a'
    'SCAPE_a'
    'DT4D_intra'
    'DT4D_inter'
    # 'SMAL_iso'
)
worker_id=$SLURM_ARRAY_TASK_ID
dataset_name=${dataset_list[$worker_id]}

echo "Testing experiment $exp_name with checkpoint $checkpoint_name" >&2
echo "Running job $worker_id: dataset_name=$dataset_name" >&2

# no smoothing
srun python test_on_dataset.py --exp_name $exp_name --dataset_name $dataset_name --checkpoint_name $checkpoint_name --data_dir $data_dir
