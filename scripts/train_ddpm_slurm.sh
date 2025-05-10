#!/bin/bash

#SBATCH -n 1
#SBATCH -t 7-00:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:8
#SBATCH --partition=mlgpu_long
#SBATCH --account=ag_ifi_laehner
#SBATCH --output=/home/s94zalek_hpc/DenoisingFunctionalMaps/SLURM_logs/train_ddpm_%j.out
#SBATCH --error=/home/s94zalek_hpc/DenoisingFunctionalMaps/SLURM_logs/train_ddpm_%j.err

# make the logs directory if it doesn't exist
mkdir -p /home/s94zalek_hpc/DenoisingFunctionalMaps/SLURM_logs

source /home/s94zalek_hpc/.bashrc
conda activate fmnet
cd /home/s94zalek_hpc/DenoisingFunctionalMaps
module load libGLU Xvfb
export PYTHONPATH=${PYTHONPATH}:/home/s94zalek_hpc/DenoisingFunctionalMaps


exp_name='ddpm_64_SMAL_sign_net_64'
dataset_name='SMAL_sign_net_64_smal'
sample_size=64
block_out_channels='64,128,128'
# block_out_channels='32,64,64'


# make directory ${dataset_name}/train in /tmp
mkdir -p /tmp/${dataset_name}
echo "Copying data to /tmp/${dataset_name}" >&2

cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/ddpm/${dataset_name}/C_1T.pt /tmp/${dataset_name}
cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/ddpm/${dataset_name}/y_T.pt /tmp/${dataset_name}
cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/ddpm/${dataset_name}/y_1.pt /tmp/${dataset_name}
cp -r /lustre/mlnvme/data/s94zalek_hpc-shape_matching/data_denoisfm/train/ddpm/${dataset_name}/config.yaml /tmp/${dataset_name}

echo "Data copied to /tmp/${dataset_name}" >&2

# sample a random integer between 0 and 1000
port=$RANDOM

srun accelerate launch --main_process_port ${port} train_ddpm.py --exp_name ${exp_name} --dataset_name ${dataset_name} --sample_size ${sample_size} --block_out_channels ${block_out_channels} 
