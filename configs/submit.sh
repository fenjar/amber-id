#!/bin/bash
#SBATCH --job-name=avatarfp
#SBATCH --partition=RTXA6000
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=100G
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

# Container Settings
CONTAINER=/netscratch/$USER/video_deanon_train_cuda.sqsh
WORKDIR=/home/$USER/video_deanon_train
DATADIR=/netscratch/$USER/tav-d/train
SCRATCH=/netscratch/$USER/avatar-fp

mkdir -p $SCRATCH

srun -K \
  --container-image=$CONTAINER \
  --container-workdir=$WORKDIR \
  --container-mounts="$WORKDIR":"$WORKDIR",$DATADIR:$DATADIR:ro,$SCRATCH:$SCRATCH \
  python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
    --data_root $DATADIR \
    --work_dir $SCRATCH \
    --F 31 --batch_identities 4 --clips_per_id 8 \
    --epochs 10 \
    --num_landmarks 80 \
    --amp --grad_accum 1 2>&1 | grep -v "inference_feedback_manager.cc"
