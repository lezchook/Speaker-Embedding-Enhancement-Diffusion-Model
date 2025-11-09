#!/bin/bash

IMAGE_NAME="seed_image"
CONTAINER_NAME="seed_container"

DATA_DIR="$(pwd)/data"
CHECKPOINTS_DIR="$(pwd)/checkpoints"
BEST_MODELS_DIR="$(pwd)/best_models"

mkdir -p "$DATA_DIR" "$CHECKPOINTS_DIR" "$BEST_MODELS_DIR"

docker build -t $IMAGE_NAME .

docker run --gpus all -it \
  --name $CONTAINER_NAME \
  -v "$DATA_DIR":/workspace/data \
  -v "$CHECKPOINTS_DIR":/workspace/checkpoints \
  -v "$BEST_MODELS_DIR":/workspace/best_models \
  $IMAGE_NAME