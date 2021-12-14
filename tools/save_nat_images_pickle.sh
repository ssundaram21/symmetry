#!/bin/bash
#SBATCH -c 2
#SBATCH --job-name=get_nat_images
#SBATCH --mem=50GB
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu

#SBATCH -D ./log/

cd /om/user/shobhita/src/symmetry/

hostname


singularity exec -B /om:/om /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/shobhita/src/symmetry/get_nat_images_as_pickle.py