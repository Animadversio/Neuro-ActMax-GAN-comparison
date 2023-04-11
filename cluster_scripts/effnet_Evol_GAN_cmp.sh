#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=73-108
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o BigGANeffnet_evol_%j.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.0 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.0 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.1 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.1 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.2 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.2 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.3 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.3 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.4 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.4 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.5 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.5 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.6 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .blocks.6 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.0 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.0 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.1 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.1 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.2 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.2 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.3 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.3 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.4 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.4 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.5 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.5 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.6 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.6 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .Linearclassifier --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6_ap --layer .Linearclassifier --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .Linearclassifier --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 0 10 --net tf_efficientnet_b6 --layer .Linearclassifier --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.0 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.0 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.1 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.1 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.2 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.2 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.3 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.3 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.4 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.4 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.5 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.5 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.6 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .blocks.6 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.0 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.0 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.1 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.1 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.2 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.2 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.3 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.3 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.4 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.4 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.5 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.5 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.6 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .blocks.6 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .Linearclassifier --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6_ap --layer .Linearclassifier --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .Linearclassifier --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 10 50 --net tf_efficientnet_b6 --layer .Linearclassifier --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.0 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.0 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.1 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.1 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.2 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.2 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.3 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.3 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.4 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.4 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.5 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.5 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.6 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .blocks.6 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.0 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.0 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.1 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.1 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.2 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.2 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.3 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.3 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.4 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.4 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.5 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.5 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.6 --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .blocks.6 --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .Linearclassifier --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6_ap --layer .Linearclassifier --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .SelectAdaptivePool2dglobal_pool --optim HessCMA CholCMA --G BigGAN --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .Linearclassifier --optim HessCMA500 CholCMA --G fc6 --rep 10
--chans 40 50 --net tf_efficientnet_b6 --layer .Linearclassifier --optim HessCMA CholCMA --G BigGAN --rep 10
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/Neuro-ActMax-GAN-comparison
python3 insilico_experiments/BigGAN_Evol_cmp_O2_cluster.py  $unit_name
