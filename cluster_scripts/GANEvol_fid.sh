#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=14-25
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o GANEvol_fid_%j.%a.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--batch_size 25 --class_id_start   0 --class_id_end 200
--batch_size 25 --class_id_start 200 --class_id_end 400
--batch_size 25 --class_id_start 400 --class_id_end 600
--batch_size 25 --class_id_start 600 --class_id_end 800
--batch_size 25 --class_id_start 800 --class_id_end 1000
--layer .AdaptiveAvgPool2davgpool --dirname resnet50_linf8_gradevol_avgpool --img_per_class 25 --class_id_start   0 --class_id_end 256
--layer .AdaptiveAvgPool2davgpool --dirname resnet50_linf8_gradevol_avgpool --img_per_class 25 --class_id_start 256 --class_id_end 512
--layer .AdaptiveAvgPool2davgpool --dirname resnet50_linf8_gradevol_avgpool --img_per_class 25 --class_id_start 512 --class_id_end 768
--layer .AdaptiveAvgPool2davgpool --dirname resnet50_linf8_gradevol_avgpool --img_per_class 25 --class_id_start 768 --class_id_end 1024
--layer .AdaptiveAvgPool2davgpool --dirname resnet50_linf8_gradevol_avgpool --img_per_class 25 --class_id_start 1024 --class_id_end 1280
--layer .AdaptiveAvgPool2davgpool --dirname resnet50_linf8_gradevol_avgpool --img_per_class 25 --class_id_start 1280 --class_id_end 1536
--layer .AdaptiveAvgPool2davgpool --dirname resnet50_linf8_gradevol_avgpool --img_per_class 25 --class_id_start 1536 --class_id_end 1792
--layer .AdaptiveAvgPool2davgpool --dirname resnet50_linf8_gradevol_avgpool --img_per_class 25 --class_id_start 1792 --class_id_end 2048
--layer .layer4 --dirname resnet50_linf8_gradevol_layer4 --img_per_class 25 --class_id_start   0 --class_id_end 256
--layer .layer4 --dirname resnet50_linf8_gradevol_layer4 --img_per_class 25 --class_id_start 256 --class_id_end 512
--layer .layer4 --dirname resnet50_linf8_gradevol_layer4 --img_per_class 25 --class_id_start 512 --class_id_end 768
--layer .layer4 --dirname resnet50_linf8_gradevol_layer4 --img_per_class 25 --class_id_start 768 --class_id_end 1024
--layer .layer4 --dirname resnet50_linf8_gradevol_layer4 --img_per_class 25 --class_id_start 1024 --class_id_end 1280
--layer .layer4 --dirname resnet50_linf8_gradevol_layer4 --img_per_class 25 --class_id_start 1280 --class_id_end 1536
--layer .layer4 --dirname resnet50_linf8_gradevol_layer4 --img_per_class 25 --class_id_start 1536 --class_id_end 1792
--layer .layer4 --dirname resnet50_linf8_gradevol_layer4 --img_per_class 25 --class_id_start 1792 --class_id_end 2048
--layer .layer3 --dirname resnet50_linf8_gradevol_layer3 --img_per_class 50 --class_id_start   0 --class_id_end 256
--layer .layer3 --dirname resnet50_linf8_gradevol_layer3 --img_per_class 50 --class_id_start 256 --class_id_end 512
--layer .layer3 --dirname resnet50_linf8_gradevol_layer3 --img_per_class 50 --class_id_start 512 --class_id_end 768
--layer .layer3 --dirname resnet50_linf8_gradevol_layer3 --img_per_class 50 --class_id_start 768 --class_id_end 1024
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
