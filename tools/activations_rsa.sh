#!/bin/bash
#SBATCH -c 2
#SBATCH --job-name=activation_mean
#SBATCH --mem=50GB
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu

#SBATCH -D ./log/

cd /om/user/shobhita/src/symmetry/

hostname


singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/shobhita/src/symmetry/main.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} \
--code_path="/om/user/shobhita/src/symmetry/" \
--output_path='/om/user/shobhita/data/symmetry/' \
--run=train \
--network=LSTM3
