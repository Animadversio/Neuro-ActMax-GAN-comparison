#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --array=1-20
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o cornet_evol_%j.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--chans  0 20 --area V1 --time_range 0 1 --G fc6    --optim HessCMA500 CholCMA --rep 5
--chans  0 20 --area V2 --time_range 0 2 --G fc6    --optim HessCMA500 CholCMA --rep 5
--chans  0 10 --area V4 --time_range 0 4 --G fc6    --optim HessCMA500 CholCMA --rep 5
--chans 10 20 --area V4 --time_range 0 4 --G fc6    --optim HessCMA500 CholCMA --rep 5
--chans  0 20 --area IT --time_range 0 2 --G fc6    --optim HessCMA500 CholCMA --rep 5
--chans  0 20 --area V1 --time_range 0 1 --G BigGAN    --optim HessCMA CholCMA --rep 5
--chans  0 20 --area V2 --time_range 0 2 --G BigGAN    --optim HessCMA CholCMA --rep 5
--chans  0 10 --area V4 --time_range 0 4 --G BigGAN    --optim HessCMA CholCMA --rep 5
--chans 10 20 --area V4 --time_range 0 4 --G BigGAN    --optim HessCMA CholCMA --rep 5
--chans  0 20 --area IT --time_range 0 2 --G BigGAN    --optim HessCMA CholCMA --rep 5
--chans 20 40 --area V1 --time_range 0 1 --G fc6    --optim HessCMA500 CholCMA --rep 5
--chans 20 40 --area V2 --time_range 0 2 --G fc6    --optim HessCMA500 CholCMA --rep 5
--chans 20 30 --area V4 --time_range 0 4 --G fc6    --optim HessCMA500 CholCMA --rep 5
--chans 30 40 --area V4 --time_range 0 4 --G fc6    --optim HessCMA500 CholCMA --rep 5
--chans 20 40 --area IT --time_range 0 2 --G fc6    --optim HessCMA500 CholCMA --rep 5
--chans 20 40 --area V1 --time_range 0 1 --G BigGAN    --optim HessCMA CholCMA --rep 5
--chans 20 40 --area V2 --time_range 0 2 --G BigGAN    --optim HessCMA CholCMA --rep 5
--chans 20 30 --area V4 --time_range 0 4 --G BigGAN    --optim HessCMA CholCMA --rep 5
--chans 30 40 --area V4 --time_range 0 4 --G BigGAN    --optim HessCMA CholCMA --rep 5
--chans 20 40 --area IT --time_range 0 2 --G BigGAN    --optim HessCMA CholCMA --rep 5
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/Neuro-ActMax-GAN-comparison
python3 insilico_experiments/CorNet_BigGAN_Evol_cmp_O2.py  $unit_name
