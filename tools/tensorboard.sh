#!/bin/bash

srun -t 2:00:00 --qos=cbmm \
singularity exec -B /om:/om /om/user/xboix/singularity/belledon-tensorflow-keras-master-latest.simg tensorboard \
--port=6058 \
--logdir=/om/user/xboix/share/minimal-images/models/
