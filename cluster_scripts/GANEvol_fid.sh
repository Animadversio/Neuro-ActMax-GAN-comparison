#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=1-5
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o GANEvol_fid_%j.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--batch_size 25 --class_id_start   0 --class_id_end 200
--batch_size 25 --class_id_start 200 --class_id_end 400
--batch_size 25 --class_id_start 400 --class_id_end 600
--batch_size 25 --class_id_start 600 --class_id_end 800
--batch_size 25 --class_id_start 800 --class_id_end 1000
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/Neuro-ActMax-GAN-comparison
python3 core/GAN_evol_sampling_O2.py  $unit_name
