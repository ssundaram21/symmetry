#!/bin/bash
#SBATCH -n 2
#SBATCH --job-name=insideness
#SBATCH --mem=1GB
#SBATCH -t 1:00:00
#SBATCH --qos=cbmm
#SBATCH --workdir=./log/

cd /om/user/xboix/src/insideness/

singularity exec -B /om:/om /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om/user/xboix/src/insideness/main.py \
--experiment_index=0 \
--host_filesystem=om \
--run=crossval_select
