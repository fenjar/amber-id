#!/bin/bash
#SBATCH --job-name=avatarfp-debug-ddp
#SBATCH --partition=A100-40GB
#SBATCH --nodes=1
#SBATCH --ntasks=2              # 2 Tasks = 2 GPUs
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

# Container Settings
CONTAINER=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh
WORKDIR=$(pwd)
DATADIR=/ds/avatars
SCRATCH=/netscratch/$USER/avatar-fp-debug-ddp

mkdir -p $SCRATCH

srun -K \
  --container-image=$CONTAINER \
  --container-workdir=$WORKDIR \
  --container-mounts="$WORKDIR":"$WORKDIR",$DATADIR:$DATADIR:ro,$SCRATCH:$SCRATCH \
  python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
    --data_root $DATADIR \
    --work_dir $SCRATCH \
    --F 51 \
    --batch_identities 2 \
    --clips_per_id 4 \
    --epochs 2 \
    --amp \
    --grad_accum 1
