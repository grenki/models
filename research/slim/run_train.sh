#!/usr/bin/env bash
DATASET_DIR=/home/grenki/src/models/dataset
TRAIN_DIR=/home/grenki/src/
CHECKPOINT_PATH=/home/grenki/Downloads/inception_v4.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --model_name=inception_v4 \
    --checkpoint_exclude_scopes=InceptionV4/Logits \
    --trainable_scopes=InceptionV4/Logits