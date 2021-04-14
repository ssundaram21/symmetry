#!/bin/bash
#SBATCH -c 2
#SBATCH --array=0-32
#SBATCH --job-name=insideness
#SBATCH --mem=16GB
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --partition=cbmm
#SBATCH -D ./log/

cd /om/user/shobhita/src/symmetry/

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om  --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/shobhita/src/symmetry/main.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} \
--host_filesystem=om-shobhita \
--run=generate_dataset \
--network=multi_lstm_init


