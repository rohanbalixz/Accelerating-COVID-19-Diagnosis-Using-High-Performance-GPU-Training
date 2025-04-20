#!/bin/bash

echo "[INFO] Starting preprocessing..."

# Activate virtual environment
source covid-env/bin/activate

# Run Python preprocessing
python src/preprocess_data.py

echo "[DONE] Preprocessing complete."

