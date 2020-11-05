#!/usr/bin/env bash
set -ex

# Training
GPU_ID=1

# Network configuration
TRAINING_DATA_PATH="/home/tanjiale/PIFu/data/training_data"
BATCH_SIZE=4
NUM_EPOCH=250

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m apps.train_shape \
    --dataroot ${TRAINING_DATA_PATH} \
    --batch_size ${BATCH_SIZE} \
    --num_epoch ${NUM_EPOCH} \
    --random_flip \
    --random_scale \
    --random_trans && \
    python -m apps.train_color \
    --dataroot ${TRAINING_DATA_PATH} \
    --batch_size ${BATCH_SIZE} \
    --num_epoch ${NUM_EPOCH} \
    --num_sample_inout 0 \
    --num_sample_color 5000 \
    --sigma 0.1 \
    --random_flip \
    --random_scale \
    --random_trans

