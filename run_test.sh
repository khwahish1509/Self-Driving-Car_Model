#!/bin/bash

# Self-Driving Car Testing Script
# This script runs the trained model in the simulator

echo "=========================================="
echo "Starting Autonomous Driving Mode"
echo "=========================================="

# Check if model exists
if [ ! -f "logs/models/model_best.h5" ]; then
    echo "Error: Trained model not found!"
    echo "Please train the model first using: ./run_train.sh"
    exit 1
fi

echo ""
echo "Instructions:"
echo "1. This will start the server on port 4567"
echo "2. Launch the Udacity simulator"
echo "3. Select 'AUTONOMOUS MODE'"
echo "4. The car should drive itself!"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="

# Run the test simulation
python -m src.test_simulation \
    --model logs/models/model_best.h5 \
    --speed 30