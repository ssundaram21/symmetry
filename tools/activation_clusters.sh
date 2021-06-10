#!/bin/bash
#SBATCH -c 4
#SBATCH --job-name=clusters
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu
#SBATCH --partition=cbmm
#SBATCH -t 10:00:00
#SBATCH -D ./log/

cd /om/user/shobhita/src/symmetry/

hostname


singularity exec -B /om:/om /om/user/xboix/singularity/xboix-tf_fujitsu3.simg \
python /om/user/shobhita/src/symmetry/activation_clustering.py
