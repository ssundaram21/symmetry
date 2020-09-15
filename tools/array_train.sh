#!/bin/bash
#SBATCH -c 2
#SBATCH --array=25-26
#SBATCH --job-name=LSTM
#SBATCH --mem=12GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 20:00:00
#SBATCH --partition=cbmm
#SBATCH -D ./log/

cd /om/user/shobhita/src/insideness/

hostname

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/shobhita/src/insideness/main.py \
--experiment_index=$((${SLURM_ARRAY_TASK_ID}+810)) \
--host_filesystem=om-shobhita \
--network=multi_lstm_init \
--run=train


