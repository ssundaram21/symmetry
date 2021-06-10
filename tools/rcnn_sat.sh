#!/bin/bash
#SBATCH -c 2
#SBATCH --job-name=rcnn
#SBATCH --mem=20GB
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu
#SBATCH --partition=cbmm

#SBATCH -D ./log/

cd /om/user/shobhita/src/symmetry/

hostname

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tf_fujitsu3.simg \
python /om/user/shobhita/src/symmetry/test_rcnn_sat.py
