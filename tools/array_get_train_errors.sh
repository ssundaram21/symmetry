#!/bin/bash
#SBATCH -c 2
#SBATCH --job-name=insideness
#SBATCH --mem=1GB
#SBATCH -t 1:00:00
#SBATCH --partition=cbmm
#SBATCH -D ./log/

cd /om/user/shobhita/src/insideness/

singularity exec -B /om:/om /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/shobhita/src/insideness/main.py \
--experiment_index=0 \
--host_filesystem=om-shobhita \
--network=multi_lstm \
--run=get_train_errors
