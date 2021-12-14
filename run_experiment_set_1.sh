#!/bin/bash
#SBATCH -c 4
#SBATCH --array=1-42
#SBATCH --job-name=exp_set_1
#SBATCH --mem=25GB
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu
#SBATCH --partition=cbmm

#SBATCH -D ./log/

cd /om/user/shobhita/src/symmetry/experiment_set_1

hostname
export CUDA_VISIBLE_DEVICES=0
/om2/user/jakubk/miniconda3/envs/torch/bin/python -c 'import torch; print(torch.rand(2,3).cuda())'

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg \
python /om/user/shobhita/src/symmetry/experiment_set_1/run_full_experiment_set_1.py --idx=${SLURM_ARRAY_TASK_ID}
