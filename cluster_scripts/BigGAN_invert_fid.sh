#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=1-8
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o BigGAN_invert_%j.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--idx_start 5000 --idx_end 7500 --batch_size 20 --max_iter 2500
--idx_start 7500 --idx_end 10000 --batch_size 20 --max_iter 2500
--idx_start 10000 --idx_end 12500 --batch_size 20 --max_iter 2500
--idx_start 12500 --idx_end 15000 --batch_size 20 --max_iter 2500
--idx_start 15000 --idx_end 17500 --batch_size 20 --max_iter 2500
--idx_start 17500 --idx_end 20000 --batch_size 20 --max_iter 2500
--idx_start 20000 --idx_end 22500 --batch_size 20 --max_iter 2500
--idx_start 22500 --idx_end 25000 --batch_size 20 --max_iter 2500
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/Neuro-ActMax-GAN-comparison
python3 core/GAN_invert_sampling_O2.py  $unit_name
