#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=75-92
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o BigGANresnet_evol_%j.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--chans 0 20 --G fc6 --net resnet50 --layer .layer1.Bottleneck1 --optim HessCMA500 --rep 10 --RFresize 1
--chans 0 20 --G fc6 --net resnet50 --layer .layer2.Bottleneck0 --optim HessCMA500 --rep 10 --RFresize 1
--chans 0 20 --G fc6 --net resnet50 --layer .layer2.Bottleneck3 --optim HessCMA500 --rep 10 --RFresize 1
--chans 0 20 --G fc6 --net resnet50 --layer .layer3.Bottleneck0 --optim HessCMA500 --rep 10 --RFresize 1
--chans 0 20 --G fc6 --net resnet50 --layer .layer3.Bottleneck2 --optim HessCMA500 --rep 10 --RFresize 1
--chans 0 20 --G fc6 --net resnet50 --layer .layer3.Bottleneck4 --optim HessCMA500 --rep 10 --RFresize 1
--chans 0 20 --G fc6 --net resnet50 --layer .layer3.Bottleneck5 --optim HessCMA500 --rep 10 --RFresize 1
--chans 0 20 --G fc6 --net resnet50 --layer .layer4.Bottleneck0 --optim HessCMA500 --rep 10 --RFresize 1
--chans 0 20 --G fc6 --net resnet50 --layer .layer4.Bottleneck2 --optim HessCMA500 --rep 10 --RFresize 1
--chans 0 20 --G BigGAN --net resnet50 --layer .layer1.Bottleneck1 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0 20 --G BigGAN --net resnet50 --layer .layer2.Bottleneck0 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0 20 --G BigGAN --net resnet50 --layer .layer2.Bottleneck3 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0 20 --G BigGAN --net resnet50 --layer .layer3.Bottleneck0 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0 20 --G BigGAN --net resnet50 --layer .layer3.Bottleneck2 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0 20 --G BigGAN --net resnet50 --layer .layer3.Bottleneck4 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0 20 --G BigGAN --net resnet50 --layer .layer3.Bottleneck5 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0 20 --G BigGAN --net resnet50 --layer .layer4.Bottleneck0 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0 20 --G BigGAN --net resnet50 --layer .layer4.Bottleneck2 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0 20 --G fc6 --net resnet50 --layer .Linearfc --optim HessCMA500  --rep 10
--chans 0 20 --G BigGAN --net resnet50 --layer .layer1.Bottleneck1 --optim HessCMA CholCMA --rep 10
--chans 0 20 --G BigGAN --net resnet50 --layer .layer2.Bottleneck0 --optim HessCMA CholCMA --rep 10
--chans 0 20 --G BigGAN --net resnet50 --layer .layer2.Bottleneck3 --optim HessCMA CholCMA --rep 10
--chans 0 20 --G BigGAN --net resnet50 --layer .layer3.Bottleneck0 --optim HessCMA CholCMA --rep 10
--chans 0 20 --G BigGAN --net resnet50 --layer .layer3.Bottleneck2 --optim HessCMA CholCMA --rep 10
--chans 0 20 --G BigGAN --net resnet50 --layer .layer3.Bottleneck4 --optim HessCMA CholCMA --rep 10
--chans 0 20 --G BigGAN --net resnet50 --layer .layer3.Bottleneck5 --optim HessCMA CholCMA --rep 10
--chans 0 20 --G BigGAN --net resnet50 --layer .layer4.Bottleneck0 --optim HessCMA CholCMA --rep 10
--chans 0 20 --G BigGAN --net resnet50 --layer .layer4.Bottleneck2 --optim HessCMA CholCMA --rep 10 
--chans 0 20 --G BigGAN --net resnet50 --layer .Linearfc --optim HessCMA CholCMA --rep 10
--chans 0 20 --G fc6 --net resnet50 --layer .layer1.Bottleneck1 --optim HessCMA500 --rep 10
--chans 0 20 --G fc6 --net resnet50 --layer .layer2.Bottleneck0 --optim HessCMA500 --rep 10
--chans 0 20 --G fc6 --net resnet50 --layer .layer2.Bottleneck3 --optim HessCMA500  --rep 10
--chans 0 20 --G fc6 --net resnet50 --layer .layer3.Bottleneck0 --optim HessCMA500 --rep 10
--chans 0 20 --G fc6 --net resnet50 --layer .layer3.Bottleneck2 --optim HessCMA500 --rep 10
--chans 0 20 --G fc6 --net resnet50 --layer .layer3.Bottleneck4 --optim HessCMA500  --rep 10
--chans 0 20 --G fc6 --net resnet50 --layer .layer3.Bottleneck5 --optim HessCMA500  --rep 10
--chans 0 20 --G fc6 --net resnet50 --layer .layer4.Bottleneck0 --optim HessCMA500 --rep 10
--chans 0 20 --G fc6 --net resnet50 --layer .layer4.Bottleneck2 --optim HessCMA500  --rep 10
--chans 0  25 --G fc6    --net resnet50_linf8 --layer .layer3.Bottleneck5 --optim HessCMA500 CholCMA  --rep 10
--chans 25 50 --G fc6    --net resnet50_linf8 --layer .layer3.Bottleneck5 --optim HessCMA500 CholCMA  --rep 10
--chans 0  25 --G BigGAN --net resnet50_linf8 --layer .layer3.Bottleneck5 --optim HessCMA CholCMA --rep 10
--chans 25 50 --G BigGAN --net resnet50_linf8 --layer .layer3.Bottleneck5 --optim HessCMA CholCMA --rep 10
--chans 0  25 --G fc6    --net resnet50_linf8 --layer .layer3.Bottleneck5 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 25 50 --G fc6    --net resnet50_linf8 --layer .layer3.Bottleneck5 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 0  25 --G BigGAN --net resnet50_linf8 --layer .layer3.Bottleneck5 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 25 50 --G BigGAN --net resnet50_linf8 --layer .layer3.Bottleneck5 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0  25 --G fc6    --net resnet50_linf8 --layer .layer1.Bottleneck1 --optim HessCMA500 CholCMA  --rep 10
--chans 0  25 --G BigGAN --net resnet50_linf8 --layer .layer1.Bottleneck1 --optim HessCMA CholCMA --rep 10
--chans 0  25 --G fc6    --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim HessCMA500 CholCMA  --rep 10
--chans 0  25 --G BigGAN --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim HessCMA CholCMA --rep 10
--chans 0  25 --G fc6    --net resnet50_linf8 --layer .layer4.Bottleneck2 --optim HessCMA500 CholCMA  --rep 10
--chans 0  25 --G BigGAN --net resnet50_linf8 --layer .layer4.Bottleneck2 --optim HessCMA CholCMA --rep 10
--chans 0  25 --G fc6    --net resnet50_linf8 --layer .Linearfc --optim HessCMA500 CholCMA  --rep 10
--chans 0  25 --G BigGAN --net resnet50_linf8 --layer .Linearfc --optim HessCMA CholCMA --rep 10
--chans 0  25 --G fc6    --net resnet50_linf8 --layer .layer1.Bottleneck1 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 0  25 --G BigGAN --net resnet50_linf8 --layer .layer1.Bottleneck1 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0  25 --G fc6    --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 0  25 --G BigGAN --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 0  25 --G fc6    --net resnet50_linf8 --layer .layer4.Bottleneck2 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 0  25 --G BigGAN --net resnet50_linf8 --layer .layer4.Bottleneck2 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 25 50 --G fc6    --net resnet50_linf8 --layer .layer1.Bottleneck1 --optim HessCMA500 CholCMA  --rep 10
--chans 25 50 --G BigGAN --net resnet50_linf8 --layer .layer1.Bottleneck1 --optim HessCMA CholCMA --rep 10
--chans 25 50 --G fc6    --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim HessCMA500 CholCMA  --rep 10
--chans 25 50 --G BigGAN --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim HessCMA CholCMA --rep 10
--chans 25 50 --G fc6    --net resnet50_linf8 --layer .layer4.Bottleneck2 --optim HessCMA500 CholCMA  --rep 10
--chans 25 50 --G BigGAN --net resnet50_linf8 --layer .layer4.Bottleneck2 --optim HessCMA CholCMA --rep 10
--chans 25 50 --G fc6    --net resnet50_linf8 --layer .Linearfc --optim HessCMA500 CholCMA  --rep 10
--chans 25 50 --G BigGAN --net resnet50_linf8 --layer .Linearfc --optim HessCMA CholCMA --rep 10
--chans 25 50 --G fc6    --net resnet50_linf8 --layer .layer1.Bottleneck1 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 25 50 --G BigGAN --net resnet50_linf8 --layer .layer1.Bottleneck1 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 25 50 --G fc6    --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 25 50 --G BigGAN --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 25 50 --G fc6    --net resnet50_linf8 --layer .layer4.Bottleneck2 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 25 50 --G BigGAN --net resnet50_linf8 --layer .layer4.Bottleneck2 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 20 50 --G fc6    --net resnet50 --layer .layer1.Bottleneck1 --optim HessCMA500 CholCMA  --rep 10
--chans 20 50 --G BigGAN --net resnet50 --layer .layer1.Bottleneck1 --optim HessCMA CholCMA --rep 10
--chans 20 50 --G fc6    --net resnet50 --layer .layer2.Bottleneck3 --optim HessCMA500 CholCMA  --rep 10
--chans 20 50 --G BigGAN --net resnet50 --layer .layer2.Bottleneck3 --optim HessCMA CholCMA --rep 10
--chans 20 50 --G fc6    --net resnet50 --layer .layer3.Bottleneck5 --optim HessCMA500 CholCMA  --rep 10
--chans 20 50 --G BigGAN --net resnet50 --layer .layer3.Bottleneck5 --optim HessCMA CholCMA --rep 10
--chans 20 50 --G fc6    --net resnet50 --layer .layer4.Bottleneck2 --optim HessCMA500 CholCMA  --rep 10
--chans 20 50 --G BigGAN --net resnet50 --layer .layer4.Bottleneck2 --optim HessCMA CholCMA --rep 10
--chans 20 50 --G fc6    --net resnet50 --layer .Linearfc --optim HessCMA500 CholCMA  --rep 10
--chans 20 50 --G BigGAN --net resnet50 --layer .Linearfc --optim HessCMA CholCMA --rep 10
--chans 20 50 --G fc6    --net resnet50 --layer .layer1.Bottleneck1 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 20 50 --G BigGAN --net resnet50 --layer .layer1.Bottleneck1 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 20 50 --G fc6    --net resnet50 --layer .layer2.Bottleneck3 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 20 50 --G BigGAN --net resnet50 --layer .layer2.Bottleneck3 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 20 50 --G fc6    --net resnet50 --layer .layer3.Bottleneck5 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 20 50 --G BigGAN --net resnet50 --layer .layer3.Bottleneck5 --optim HessCMA CholCMA --rep 10 --RFresize 1
--chans 20 50 --G fc6    --net resnet50 --layer .layer4.Bottleneck2 --optim HessCMA500 CholCMA  --rep 10 --RFresize 1
--chans 20 50 --G BigGAN --net resnet50 --layer .layer4.Bottleneck2 --optim HessCMA CholCMA --rep 10 --RFresize 1
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
