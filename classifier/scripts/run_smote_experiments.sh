#!/bin/bash

# SMOTE Experiments Script
# 5 backbone models × 2 augmentation levels = 10 experiments total

echo "Starting SMOTE experiments..."
echo "Dataset: adni_ct"
echo "Models: svm, mlp, mlp-a, gcn, gat"
echo "Augmentation levels: min, full"
echo "Total experiments: 10"
echo "================================"

# Array of models and augmentation levels
models=("svm" "mlp" "mlp-a" "gcn" "gat")
aug_levels=("min" "full")

# Function to run experiment
run_experiment() {
    local model=$1
    local aug_level=$2
    local exp_num=$3
    
    echo ""
    echo "[$exp_num/10] Running experiment: $model with SMOTE_$aug_level"
    echo "Command: python main.py --data adni_ct --model $model --augmentation SMOTE --aug_level $aug_level"
    echo "Started at: $(date)"
    
    # Run the experiment
    python main.py --data adni_ct --model $model --augmentation SMOTE --aug_level $aug_level
    
    # Check if experiment was successful
    if [ $? -eq 0 ]; then
        echo "✓ Experiment $exp_num completed successfully"
    else
        echo "✗ Experiment $exp_num failed"
    fi
    
    echo "Finished at: $(date)"
    echo "================================"
}

# Run all experiments
exp_counter=1

for model in "${models[@]}"; do
    for aug_level in "${aug_levels[@]}"; do
        run_experiment "$model" "$aug_level" "$exp_counter"
        exp_counter=$((exp_counter + 1))
        
        # Add a small delay between experiments
        sleep 2
    done
done

echo ""
echo "All SMOTE experiments completed!"
echo "Results saved in:"
echo "  - Individual experiment logs: ./logs/"
echo "  - Summary Excel file: /home/user14/bagle/summary/experiment_summary.xlsx"
echo "  - WandB project: adni_ct_<model>"
