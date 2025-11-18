#!/bin/bash
#SBATCH --job-name=avatarfp-stgcn-tavd-g3-head-ce
#SBATCH --partition=H100-SLT
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=100G
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=12
#SBATCH --time=20:00:00
#SBATCH --output=slurm-%j.out

# Container Settings
CONTAINER=/netscratch/$USER/video_deanon_train_cuda.sqsh
WORKDIR=/home/$USER/video_deanon_train
DATADIR=/netscratch/$USER
SCRATCH=/netscratch/$USER/tavd_g3_ce

mkdir -p $SCRATCH

srun -K \
  --container-image=$CONTAINER \
  --container-workdir=$WORKDIR \
  --container-mounts="$WORKDIR":"$WORKDIR",$DATADIR:$DATADIR:ro,$SCRATCH:$SCRATCH \
  bash -lc "\
  set -euo pipefail; \
  echo '[STEP] starting training...'; \
  python stgcndual.py \
    --data_root $SCRATCH/g3_train_landmarks_npy \
    --save_dir $SCRATCH/model \
    --num_epochs 3000 \
    --warmup_epochs 3000 \
    --embedding_size 256 \
    --amp \
    --batch_p 8 \
    --batch_k 8 \
    --no_motion \
  "