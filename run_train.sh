#!/bin/bash

# Self-Driving Car Training Script
# This script trains the CNN model on collected driving data

echo "=========================================="
echo "Starting Self-Driving Car Model Training"
echo "=========================================="

# Check if processed data exists
if [ ! -f "data/processed/merged_balanced.csv" ]; then
    echo "Error: Processed data not found!"
    echo "Please run data preprocessing first:"
    echo "  python -m src.utils --merge_csvs data/raw --out data/processed/merged.csv --balance"
    exit 1
fi

# Train the model
python -m src.train \
    --csv data/processed/merged_balanced.csv \
    --batch 32 \
    --epochs 20 \
    --model_out logs/models/model_best.h5

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Model saved to: logs/models/model_best.h5"
echo "Training plot: logs/training_plots/loss.png"
echo "=========================================="