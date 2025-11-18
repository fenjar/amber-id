#!/bin/bash
#SBATCH --job-name=avatar-preprocessing
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --partition=RTXA6000
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

# Container Settings
CONTAINER=/netscratch/fschulz/video_deanon_train_cuda.sqsh

srun -K \
  --container-image=$CONTAINER \
  --container-workdir=/home/fschulz/video_deanon_train \
  --container-mounts=/home/fschulz:/home/fschulz,/netscratch/fschulz:/netscratch/fschulz \
  bash -c "python -m torch.distributed.run --standalone --nproc_per_node=2 train.py --data_root /netscratch/fschulz/tav-d/train --work_dir /netscratch/fschulz/lm126-f71 --F 71 --batch_identities 8 --clips_per_id 16 --preprocessing_only --num_landmarks 126  2>&1 | grep -v "inference_feedback_manager.cc""