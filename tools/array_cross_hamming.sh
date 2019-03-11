#!/bin/bash
#SBATCH -n 2
#SBATCH --array=40
#SBATCH --job-name=insideness
#SBATCH --mem=16GB
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

cd /om/user/xboix/src/insideness/

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om  --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om/user/xboix/src/insideness/main.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} \
--host_filesystem=om \
--run=cross_dataset_hamming \
--network=crossing


