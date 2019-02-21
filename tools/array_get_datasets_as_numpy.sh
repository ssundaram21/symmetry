#!/bin/bash
#SBATCH -n 2
#SBATCH --array=41,43,45,47,49
#SBATCH --job-name=insideness
#SBATCH --mem=8GB
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

cd /om/user/xboix/src/insideness/

singularity exec -B /om:/om  --nv /om/user/xboix/singularity/xboix-tensorflow.simg \
python /om/user/xboix/src/insideness/main.py \
--experiment_index=${SLURM_ARRAY_TASK_ID} \
--host_filesystem=om \
--network=crossing \
--run=get_dataset_as_numpy


