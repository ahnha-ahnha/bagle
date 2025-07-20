#!/usr/bin/env python3
"""
Simple test script for ADNI_CT data loading and basic SVM training
"""

import torch
import numpy as np
from utils.metric import *
from utils.loader import *
from utils.model import *
from utils.train import *
import os
from collections import Counter

def load_real_pt_data(data_type):
    """Load real.pt data for specified ADNI dataset type"""
    data_paths = {
        'adni_ct': '/home/user14/bagle/data/ADNI_CT/real.pt',
        'adni_amy': '/home/user14/bagle/data/ADNI_Amy/real.pt',
        'adni_fdg': '/home/user14/bagle/data/ADNI_FDG/real.pt',
        'adni_tau': '/home/user14/bagle/data/ADNI_Tau/real.pt'
    }
    
    data_path = data_paths[data_type]
    print(f"Loading data from: {data_path}")
    
    data = torch.load(data_path)
    
    A = data['A']  # Adjacency matrices
    X = data['X']  # Features
    y = data['Y']  # Labels
    eigenvalues = data['EIGVAL']  # Eigenvalues
    eigenvectors = data['EIGVEC']  # Eigenvectors
    ptids = data['PTID']  # Patient IDs
    ages = data['AGE']  # Ages
    sex = data['SEX']  # Sex
    folds = data['fold']  # Fold information
    
    print(f"Data loaded successfully:")
    print(f"  Samples: {A.shape[0]}")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Unique labels: {torch.unique(y)}")
    print(f"  Fold values: {torch.unique(folds)}")
    
    return A, X, y, eigenvalues, eigenvectors, ptids, ages, sex, folds

def create_fold_splits_from_existing(folds, n_folds=5):
    """Create train/test splits based on existing fold information"""
    if isinstance(folds, torch.Tensor):
        folds = folds.numpy()
    
    fold_splits = []
    
    for fold_num in range(n_folds):
        test_idx = np.where(folds == fold_num)[0]
        train_idx = np.where(folds != fold_num)[0]
        fold_splits.append((train_idx, test_idx))
        print(f"Fold {fold_num}: {len(train_idx)} train samples, {len(test_idx)} test samples")
    
    return fold_splits

def test_basic_functionality():
    """Test basic functionality without WandB"""
    
    # Load data
    A, X, y, eigenvalues, eigenvectors, ptids, ages, sex, folds = load_real_pt_data('adni_ct')
    
    # Create fold splits
    fold_splits = create_fold_splits_from_existing(folds, n_folds=5)
    
    # Test first fold
    train_idx, test_idx = fold_splits[0]
    
    print(f"\nTesting Fold 0:")
    print(f"Train indices: {len(train_idx)}")
    print(f"Test indices: {len(test_idx)}")
    
    # Test data shapes
    train_X = X[train_idx]
    train_y = y[train_idx]
    test_X = X[test_idx]
    test_y = y[test_idx]
    
    print(f"Train X shape: {train_X.shape}")
    print(f"Train y shape: {train_y.shape}")
    print(f"Test X shape: {test_X.shape}")
    print(f"Test y shape: {test_y.shape}")
    
    # Test label distribution
    train_counter = Counter(train_y.numpy())
    test_counter = Counter(test_y.numpy())
    
    print(f"Train label distribution: {train_counter}")
    print(f"Test label distribution: {test_counter}")
    
    print("\nBasic functionality test completed successfully!")

if __name__ == "__main__":
    test_basic_functionality()
