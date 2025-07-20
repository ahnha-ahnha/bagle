#!/bin/bash

# Quick test script for LU generator on ADNI_Amy dataset only
echo "Starting LU Generator Test on ADNI_Amy..."
echo "========================================="

GENERATOR="MLP"
DATASET="ADNI_Amy"

# Test with just one fold to verify everything works
echo "Testing $GENERATOR on $DATASET, fold 0"

python main.py \
    --data_path "/home/user14/bagle/data/$DATASET" \
    --dataset_name "$DATASET" \
    --generator "$GENERATOR" \
    --fold 0 \
    --epoch 100 \
    --batch 16 \
    --lr 0.001 \
    --hidden_dim 128 \
    --aug_strategy "LU" \
    --seed 42 \
    --device 0 \
    --is_save True

echo "Test completed!"
echo "Check /home/user14/bagle/data/$DATASET/syn/ for generated data"
