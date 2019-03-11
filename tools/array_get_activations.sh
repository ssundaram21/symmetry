#!/bin/bash
#SBATCH -n 2
#SBATCH --array=1
#SBATCH --job-name=insideness
#SBATCH --mem=32GB
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH -t 2:00:00
#SBATCH --qos=cbmm
#SBATCH --workdir=./log/

cd /om/user/xboix/src/insideness/

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om/user/xboix/src/insideness/main.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} \
--host_filesystem=om \
--network=dilation \
--run=extract_activations
