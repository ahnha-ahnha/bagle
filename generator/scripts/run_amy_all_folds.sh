#!/bin/bash

# LU generator for ADNI_Amy dataset, all folds
echo "Starting LU Generator for ADNI_Amy - All Folds"
echo "=============================================="

DATASET="ADNI_Amy"
GENERATOR="MLP"
FOLDS=(0 1 2 3 4)

# Loop through each fold
for fold in "${FOLDS[@]}"; do
    echo "Training fold: $fold"
    
    python main.py \
        --data_path "/home/user14/bagle/data/$DATASET" \
        --dataset_name "$DATASET" \
        --generator "$GENERATOR" \
        --fold "$fold" \
        --epoch 200 \
        --batch 16 \
        --lr 0.001 \
        --hidden_dim 128 \
        --aug_strategy "LU" \
        --seed 42 \
        --device 0 \
        --is_save True
    
    echo "Completed fold $fold"
    echo "-------------------"
done

echo "Completed all folds for $GENERATOR on $DATASET"
echo "Synthetic data saved in /home/user14/bagle/data/$DATASET/syn/"
ls -la "/home/user14/bagle/data/$DATASET/syn/"
