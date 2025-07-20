#!/bin/bash

# Generator baseline script for 5-fold cross validation
# This script will train generators for each fold and dataset

DATASETS=("ADNI_Amy" "ADNI_CT" "ADNI_FDG" "ADNI_Tau")
GENERATORS=("MLP" "TabNet" "FT" "NODE")
FOLDS=(0 1 2 3 4)

echo "Starting Generator Baseline Training..."
echo "======================================="

# Loop through each dataset
for dataset in "${DATASETS[@]}"; do
    echo "Processing dataset: $dataset"
    
    # Loop through each generator type
    for generator in "${GENERATORS[@]}"; do
        echo "  Using generator: $generator"
        
        # Loop through each fold
        for fold in "${FOLDS[@]}"; do
            echo "    Training fold: $fold"
            
            python main.py \
                --data_path "/home/user14/bagle/data/$dataset" \
                --dataset_name "$dataset" \
                --generator "$generator" \
                --fold "$fold" \
                --epoch 500 \
                --batch 16 \
                --lr 0.001 \
                --hidden_dim 128 \
                --aug_strategy "LU" \
                --seed 42 \
                --device 0 \
                --is_save True
            
            echo "    Completed fold $fold for $generator on $dataset"
        done
        
        echo "  Completed all folds for $generator on $dataset"
    done
    
    echo "Completed all generators for $dataset"
    echo "------------------------------------"
done

echo "All generator training completed!"
echo "Synthetic data saved in respective syn/ folders"
