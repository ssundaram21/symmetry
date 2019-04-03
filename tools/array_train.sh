#!/bin/bash
#SBATCH -n 1
#SBATCH --array=0-3
#SBATCH -c 1
#SBATCH --job-name=dilation
#SBATCH --mem=12GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -t 20:00:00
#SBATCH --qos=cbmm
#SBATCH --workdir=./log/

cd /om/user/xboix/src/insideness/

hostname

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om/user/xboix/src/insideness/main.py \
--experiment_index=$((${SLURM_ARRAY_TASK_ID} + 0)) \
--host_filesystem=om \
--network=multi_lstm_init \
--run=train \
--error_correction


