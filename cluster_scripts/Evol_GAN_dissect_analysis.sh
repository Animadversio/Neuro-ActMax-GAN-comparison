#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=1-10
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o evol_dissect_%j.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--chans 0 10 --net resnet50 --layer .layer3.Bottleneck5
--chans 0 10 --net resnet50 --layer .layer4.Bottleneck0
--chans 0 10 --net resnet50 --layer .layer4.Bottleneck2
--chans 0 10 --net resnet50 --layer .Linearfc
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.5
--chans 0 10 --net tf_efficientnet_b6 --layer .blocks.6
--chans 0 10 --net resnet50_linf8 --layer .layer3.Bottleneck5
--chans 0 10 --net resnet50_linf8 --layer .layer4.Bottleneck0
--chans 0 10 --net resnet50_linf8 --layer .layer4.Bottleneck2
--chans 0 10 --net resnet50_linf8 --layer .Linearfc
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/Neuro-ActMax-GAN-comparison
python3 insilico_experiments/BigGAN_FeatCov_lowmem_O2.py  $unit_name
