#!/usr/bin/env bash
set -ex

# Training
GPU_ID=0

# Network configuration
# TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/rp_twindom"
# TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/rp_dataset"
# TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/single_nba_dataset"
# TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/single_nba_2ku_dataset"
TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/nba_dataset/rest_pose"
# TRAINING_DATA_PATH="/home/tanjiale/pifu/training_data/nba_dataset/both"
BATCH_SIZE=4
NUM_EPOCH=30
LR=0.0001
name="nba_rest_pose_sigma_0.0275"

# Training configuration
nba_sigma=0.0275
twindom_sigma=5
rp_sigma=0.005
# --resume_epoch 39 \
# --continue_train

# command
python -m apps.train_shape \
    --gpu_id ${GPU_ID} \
    --name ${name} \
    --dataroot ${TRAINING_DATA_PATH} \
    --batch_size ${BATCH_SIZE} \
    --num_sample_inout 5000 \
    --num_epoch ${NUM_EPOCH} \
    --learning_rate ${LR} \
    --sigma ${nba_sigma}
    # --random_flip \
    # --random_scale \
    # --random_trans
