#!/bin/bash
#SBATCH -t 6:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=40G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 4-19%8
#SBATCH -o Evol_BG_LRM_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e Evol_BG_LRM_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--layershort conv1_relu --rep 5 --channel_rng 0 25 --RFresize
--layershort conv2_relu --rep 5 --channel_rng 0 25 --RFresize
--layershort conv3_relu --rep 5 --channel_rng 0 25 --RFresize
--model alexnet_lrm2 --layershort conv1_relu --rep 5 --channel_rng 0 25 --RFresize
--model alexnet_lrm2 --layershort conv2_relu --rep 5 --channel_rng 0 25 --RFresize
--model alexnet_lrm2 --layershort conv3_relu --rep 5 --channel_rng 0 25 --RFresize
--model alexnet_lrm2 --layershort conv4_relu --rep 5 --channel_rng 0 25
--model alexnet_lrm2 --layershort conv5_relu --rep 5 --channel_rng 0 25
--model alexnet_lrm2 --layershort fc6_relu --rep 5 --channel_rng 0 25 
--model alexnet_lrm2 --layershort fc7_relu --rep 5 --channel_rng 0 25 
--model alexnet_lrm2 --layershort fc8 --rep 5 --channel_rng 0 25
--model alexnet_lrm1 --layershort conv1_relu --rep 5 --channel_rng 0 25 --RFresize
--model alexnet_lrm1 --layershort conv2_relu --rep 5 --channel_rng 0 25 --RFresize
--model alexnet_lrm1 --layershort conv3_relu --rep 5 --channel_rng 0 25 --RFresize
--model alexnet_lrm1 --layershort conv4_relu --rep 5 --channel_rng 0 25
--model alexnet_lrm1 --layershort conv5_relu --rep 5 --channel_rng 0 25
--model alexnet_lrm1 --layershort fc6_relu --rep 5 --channel_rng 0 25 
--model alexnet_lrm1 --layershort fc7_relu --rep 5 --channel_rng 0 25 
--model alexnet_lrm1 --layershort fc8 --rep 5 --channel_rng 0 25
'
# --layershort conv1_relu --rep 5 --channel_rng 0 25 
# --layershort conv2_relu --rep 5 --channel_rng 0 25 
# --layershort conv3_relu --rep 5 --channel_rng 0 25 
# --layershort conv4_relu --rep 5 --channel_rng 0 25
# --layershort conv5_relu --rep 5 --channel_rng 0 25
# --layershort fc6_relu --rep 5 --channel_rng 0 25 
# --layershort fc7_relu --rep 5 --channel_rng 0 25 
# --layershort fc8 --rep 5 --channel_rng 0 25

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

# load modules
module load python
mamba deactivate
# module load cuda cudnn
mamba activate torch
which python

# run code
cd /n/home12/binxuwang/Github/Neuro-ActMax-GAN-comparison
python insilico_experiments/BigGAN_Evol_cmp_LRM_FAS_cluster.py  $param_name
