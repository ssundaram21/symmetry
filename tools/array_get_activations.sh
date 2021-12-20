#!/bin/bash
#SBATCH -c 2
#SBATCH --array=54
#SBATCH --job-name=DIL_54_act
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
--code_path="/om/user/shobhita/src/symmetry/" \
--output_path='om-shobhita': '/om/user/shobhita/data/symmetry/' \
--run=extract_activations \
--network=LSTM3
