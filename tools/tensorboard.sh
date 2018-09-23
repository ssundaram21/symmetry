#!/bin/bash
#SBATCH -n 2
#SBATCH --job-name=insideness
#SBATCH --mem=2GB
#SBATCH -t 10:00:00
#SBATCH --qos=cbmm
#SBATCH --workdir=./log


cd /om/user/xboix/src/insideness/


singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow.simg tensorboard \
--port=6058 \
--logdir=/om/user/xboix/share/insideness/
