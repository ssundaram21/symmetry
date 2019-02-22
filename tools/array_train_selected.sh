#!/bin/bash
#SBATCH -n1
#SBATCH --array=0-999
#SBATCH --exclude=node030
#SBATCH --job-name=insideness
#SBATCH --mem=12GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 10:00:00
#SBATCH --workdir=./log/

cd /om/user/xboix/src/insideness/

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om/user/xboix/src/insideness/main.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} \
--host_filesystem=om \
--network=unet \
--run=train_selected
