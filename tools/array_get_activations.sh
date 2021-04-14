#!/bin/bash
#SBATCH -c 2
#SBATCH --array=654
#SBATCH --job-name=activations_654
#SBATCH --mem=160GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 5:00:00
#SBATCH --partition=cbmm
#SBATCH -D ./log/

cd /om/user/shobhita/src/insideness/

hostname

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/shobhita/src/insideness/main.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} \
--host_filesystem=om-shobhita \
--network=multi_lstm_init \
--run=extract_activations
