#!/bin/bash
#SBATCH -c 2
#SBATCH -n 1
#SBATCH --array=0-630
#SBATCH --job-name=MultiLSTM
#SBATCH --mem=12GB
#SBATCH --gres=gpu:titan-x:1
#SBATCH -t 5:00:00
#SBATCH --partition=normal
#SBATCH -D ./log/

cd /om/user/shobhita/src/insideness/

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /om/user/shobhita/singularity/xboix-tensorflow1.14.simg \
python /om/user/shobhita/src/insideness/main.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} \
--host_filesystem=om-shobhita \
--network=multi_lstm_init \
--run=evaluate_generalization
