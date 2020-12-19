#!/usr/bin/env bash
set -ex

# Training
GPU_ID=0

# Network configuration
# TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/rp_twindom"
# TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/rp_dataset"
# TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/single_nba_dataset"
TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/nba_dataset/rest_pose"
BATCH_SIZE=4
NUM_EPOCH=35
LR=0.001
name="nba_single_dataset_updated_bounding"

# Training configuration
nba_sigma=0.025
twindom_sigma=5
rp_sigma=0.005
# --resume_epoch 9 \
# --continue_train \

# command
python -m apps.train_color \
  --name ${name} \
  --gpu_id ${GPU_ID} \
  --dataroot ${TRAINING_DATA_PATH} \
  --batch_size ${BATCH_SIZE} \
  --num_epoch ${NUM_EPOCH} \
  --num_sample_inout 0 \
  --num_sample_color 5000 \
  --sigma ${nba_sigma} \
  --norm_color "group" \
  --random_flip \
  --random_scale \
  --random_trans

