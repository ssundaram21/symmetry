#!/bin/bash
#SBATCH -c 2
#SBATCH --job-name=symmetry
#SBATCH --mem=1GB
#SBATCH -t 10:00:00
#SBATCH --partition=cbmm
#SBATCH -D ./log/

cd /om/user/shobhita/src/symmetry/data/

singularity exec -B /om:/om /om/user/xboix/singularity/xboix-tensorflow1.14.simg \

python /om/user/shobhita/src/symmetry/data/symmetry_images.py 

