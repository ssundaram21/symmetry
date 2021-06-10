#!/bin/bash
#SBATCH -c 2
#SBATCH --array=90
#SBATCH --job-name=LSTM3_90_act
#SBATCH --mem=160GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu
#SBATCH --partition=normal
#SBATCH -t 5:00:00
#SBATCH -D ./log/

cd /om/user/shobhita/src/symmetry/

hostname

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/shobhita/src/symmetry/main.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} \
--host_filesystem=om-shobhita \
--network=LSTM3 \
--run=extract_activations
