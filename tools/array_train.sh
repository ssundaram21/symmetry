#!/bin/bash
#SBATCH -c 2
#SBATCH --array=90
#SBATCH --job-name=lstm_train
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu
#SBATCH --partition=cbmm
#SBATCH -t 20:00:00
#SBATCH -D ./log/

cd /om/user/shobhita/src/symmetry/

hostname

/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/shobhita/src/symmetry/main.py \
--experiment_index=90 \
--code_path="/om/user/shobhita/src/symmetry/" \
--output_path='/om/user/shobhita/data/symmetry/' \
--run=train \
--network=LSTM3


