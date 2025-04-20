#!/bin/bash

# Default training parameters
EPOCHS=10
BATCH_SIZE=32
USE_GPU=true

echo "[INFO] Starting training job..."
echo "[CONFIG] Epochs: $EPOCHS, Batch Size: $BATCH_SIZE, GPU: $USE_GPU"

# Activate your conda or virtual environment (modify if using venv)
# For Conda
# source activate covid-env

# For venv (Mac local)
source covid-env/bin/activate

# Run the Python training script
if [ "$USE_GPU" = true ]; then
    python src/train_model.py --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --gpu
else
    python src/train_model.py --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"
fi

echo "[DONE] Training finished."

