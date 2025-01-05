#!/bin/bash
#SBATCH -t 6:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100      # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=50G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 1-4
#SBATCH -o /n/home12/binxuwang/Github/Neuro-ActMax-GAN-comparison/fasrc_logs/GAN_Hessian_%A_%a.out  
#SBATCH -e /n/home12/binxuwang/Github/Neuro-ActMax-GAN-comparison/fasrc_logs/GAN_Hessian_%A_%a.err  
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--start 0 --end 250 --dist MSE --GAN fc6
--start 250 --end 500 --dist MSE --GAN fc6
--start 500 --end 750 --dist MSE --GAN fc6
--start 750 --end 1000 --dist MSE --GAN fc6'

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"


module load python
conda deactivate
mamba activate torch2
which python
which python3

cd ~/Github/Neuro-ActMax-GAN-comparison
python3 insilico_experiments/GAN_Hessian_mass_compute.py $param_name 