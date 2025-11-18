#!/bin/bash
#SBATCH --job-name=avatarfp-debug
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

# Container Settings
CONTAINER=/netscratch/fschulz/video_deanon_train.sqsh
DATADIR=/netscratch/$USER/tav-d-debug
SCRATCH=/netscratch/$USER/avatar-fp-debug

mkdir -p $SCRATCH

srun -K \
  --container-image=$CONTAINER \
  --container-workdir=/home/fschulz/video_deanon_train \
  --container-mounts=/home/fschulz:/home/fschulz,/netscratch/fschulz:/netscratch/fschulz \
  bash -c "MASTER_ADDR=localhost MASTER_PORT=12355 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 python train.py \
    --data_root $DATADIR \
    --work_dir $SCRATCH \
    --F 31 \
    --batch_identities 2 \
    --clips_per_id 4 \
    --epochs 2 \
    --amp \
    --grad_accum 1"
