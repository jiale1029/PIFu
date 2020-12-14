#!/usr/bin/env bash
set -ex

# Training
GPU_ID=0

# Network configuration
# TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/rp_twindom"
# TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/rp_dataset"
TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/nba_dataset/rest_pose"
BATCH_SIZE=4
NUM_EPOCH=200
LR=0.001
name="nba_rest_pose_normalized"

# Training configuration
nba_sigma=0.005
twindom_sigma=5
rp_sigma=0.005

# command
python -m apps.train_shape \
    --gpu_id ${GPU_ID} \
    --name ${name} \
    --dataroot ${TRAINING_DATA_PATH} \
    --batch_size ${BATCH_SIZE} \
    --num_epoch ${NUM_EPOCH} \
    --learning_rate ${LR} \
    --sigma ${nba_sigma} \
    --random_flip \
    --random_scale \
    --random_trans && \
    python -m apps.train_color \
    --name ${name} \
    --gpu_id ${GPU_ID} \
    --dataroot ${TRAINING_DATA_PATH} \
    --batch_size ${BATCH_SIZE} \
    --num_epoch ${NUM_EPOCH} \
    --num_sample_inout 0 \
    --num_sample_color 5000 \
    --norm_color "group" \
    --sigma 0.1 \
    --random_flip \
    --random_scale \
    --random_trans

