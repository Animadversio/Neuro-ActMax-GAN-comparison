#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=25-48
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o gradBigGANresnet_evol_%j.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--chans 0 10 --G BigGAN --net resnet50 --layer .layer1 --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10 --RFresize 1
--chans 0 10 --G BigGAN --net resnet50 --layer .layer2 --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10 --RFresize 1
--chans 0 10 --G BigGAN --net resnet50 --layer .layer2 --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10
--chans 0 10 --G BigGAN --net resnet50 --layer .layer3 --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10 --RFresize 1
--chans 0 10 --G BigGAN --net resnet50 --layer .layer4 --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10
--chans 0 10 --G BigGAN --net resnet50 --layer .Linearfc --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10
--chans 0 10 --G BigGAN --net resnet50_linf8 --layer .layer1 --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10 --RFresize 1
--chans 0 10 --G BigGAN --net resnet50_linf8 --layer .layer2 --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10 --RFresize 1
--chans 0 10 --G BigGAN --net resnet50_linf8 --layer .layer2 --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10
--chans 0 10 --G BigGAN --net resnet50_linf8 --layer .layer3 --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10 --RFresize 1
--chans 0 10 --G BigGAN --net resnet50_linf8 --layer .layer4 --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10
--chans 0 10 --G BigGAN --net resnet50_linf8 --layer .Linearfc --optim Adam001 Adam001Hess Adam0003 Adam0003Hess Adam0001 Adam0001Hess SGD001 SGD001Hess SGD0003 SGD0003Hess SGD0001 SGD0001Hess --rep 10
--chans 0 10 --G fc6 --net resnet50 --layer .layer1 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 10 --G fc6 --net resnet50 --layer .layer2 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 10 --G fc6 --net resnet50 --layer .layer2 --optim Adam01 Adam01Hess --rep 10 --batch 10
--chans 0 10 --G fc6 --net resnet50 --layer .layer3 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 10 --G fc6 --net resnet50 --layer .layer4 --optim Adam01 Adam01Hess --rep 10 --batch 10
--chans 0 10 --G fc6 --net resnet50 --layer .Linearfc --optim Adam01 Adam01Hess --rep 10 --batch 10
--chans 0 10 --G fc6 --net resnet50_linf8 --layer .layer1 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 10 --G fc6 --net resnet50_linf8 --layer .layer2 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 10 --G fc6 --net resnet50_linf8 --layer .layer2 --optim Adam01 Adam01Hess --rep 10 --batch 10
--chans 0 10 --G fc6 --net resnet50_linf8 --layer .layer3 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 10 --G fc6 --net resnet50_linf8 --layer .layer4 --optim Adam01 Adam01Hess --rep 10 --batch 10
--chans 0 10 --G fc6 --net resnet50_linf8 --layer .Linearfc --optim Adam01 Adam01Hess --rep 10 --batch 10
--chans 0 25 --G BigGAN --net resnet50 --layer .layer1.Bottleneck1 --optim Adam001 Adam001Hess --rep 10 --RFresize 1
--chans 0 25 --G BigGAN --net resnet50 --layer .layer2.Bottleneck3 --optim Adam001 Adam001Hess --rep 10 --RFresize 1
--chans 0 25 --G BigGAN --net resnet50 --layer .layer2.Bottleneck3 --optim Adam001 Adam001Hess --rep 10
--chans 0 25 --G BigGAN --net resnet50 --layer .layer3.Bottleneck5 --optim Adam001 Adam001Hess --rep 10 --RFresize 1
--chans 0 25 --G BigGAN --net resnet50 --layer .layer4.Bottleneck2 --optim Adam001 Adam001Hess --rep 10
--chans 0 25 --G BigGAN --net resnet50 --layer .Linearfc --optim Adam001 Adam001Hess --rep 10
--chans 0 25 --G BigGAN --net resnet50_linf8 --layer .layer1.Bottleneck1 --optim Adam001 Adam001Hess --rep 10 --RFresize 1
--chans 0 25 --G BigGAN --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim Adam001 Adam001Hess --rep 10 --RFresize 1
--chans 0 25 --G BigGAN --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim Adam001 Adam001Hess --rep 10
--chans 0 25 --G BigGAN --net resnet50_linf8 --layer .layer3.Bottleneck5 --optim Adam001 Adam001Hess --rep 10 --RFresize 1
--chans 0 25 --G BigGAN --net resnet50_linf8 --layer .layer4.Bottleneck2 --optim Adam001 Adam001Hess --rep 10
--chans 0 25 --G BigGAN --net resnet50_linf8 --layer .Linearfc --optim Adam001 Adam001Hess --rep 10
--chans 0 25 --G fc6 --net resnet50 --layer .layer1.Bottleneck1 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 25 --G fc6 --net resnet50 --layer .layer2.Bottleneck3 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 25 --G fc6 --net resnet50 --layer .layer2.Bottleneck3 --optim Adam01 Adam01Hess --rep 10 --batch 10
--chans 0 25 --G fc6 --net resnet50 --layer .layer3.Bottleneck5 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 25 --G fc6 --net resnet50 --layer .layer4.Bottleneck2 --optim Adam01 Adam01Hess --rep 10 --batch 10
--chans 0 25 --G fc6 --net resnet50 --layer .Linearfc --optim Adam01 Adam01Hess --rep 10 --batch 10
--chans 0 25 --G fc6 --net resnet50_linf8 --layer .layer1.Bottleneck1 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 25 --G fc6 --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 25 --G fc6 --net resnet50_linf8 --layer .layer2.Bottleneck3 --optim Adam01 Adam01Hess --rep 10 --batch 10
--chans 0 25 --G fc6 --net resnet50_linf8 --layer .layer3.Bottleneck5 --optim Adam01 Adam01Hess --rep 10 --batch 10 --RFresize 1
--chans 0 25 --G fc6 --net resnet50_linf8 --layer .layer4.Bottleneck2 --optim Adam01 Adam01Hess --rep 10 --batch 10
--chans 0 25 --G fc6 --net resnet50_linf8 --layer .Linearfc --optim Adam01 Adam01Hess --rep 10 --batch 10
'


export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/Neuro-ActMax-GAN-comparison
python3 insilico_experiments/BigGAN_gradEvol_cmp_O2_cluster.py  $unit_name
