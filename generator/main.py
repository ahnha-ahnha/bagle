import os
import random
import argparse
import numpy as np
import torch
from collections import Counter
from imblearn.over_sampling import SMOTE

def create_fold_data(data_dict, fold_idx):
    """Create training data for a specific fold"""
    fold_mask = data_dict['fold'] == fold_idx
    
    # Get training data (all folds except current one)
    train_mask = ~fold_mask
    
    X_train = data_dict['X'][train_mask].squeeze()  # Remove the last dimension if it's 1
    Y_train = data_dict['Y'][train_mask]
    PTID_train = data_dict['PTID'][train_mask]
    
    return X_train, Y_train, PTID_train

def smote_augmentation_min(X, Y, ptids):
    """
    SMOTE augmentation - min setting: balance minority class to majority class count
    Returns only the augmented samples (not including original data)
    
    Args:
        X: Input features [batch_size, feature_dim]
        Y: Labels [batch_size]
        ptids: Patient IDs [batch_size] (not used, kept for compatibility)
    
    Returns:
        X_aug: Only augmented features (original data excluded)
        Y_aug: Only augmented labels (original data excluded)
        ptids_aug: None (SMOTE samples don't have corresponding PTIDs)
    """
    # Convert to numpy for SMOTE
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    # Apply SMOTE only on X features
    smote = SMOTE(random_state=0)
    X_combined_np, Y_combined_np = smote.fit_resample(X_np, Y_np)
    
    # Extract only the augmented samples (exclude original data)
    original_size = len(X_np)
    X_aug_np = X_combined_np[original_size:]
    Y_aug_np = Y_combined_np[original_size:]
    
    # Convert back to torch tensors
    X_aug = torch.tensor(X_aug_np, dtype=X.dtype, device=X.device)
    Y_aug = torch.tensor(Y_aug_np, dtype=Y.dtype, device=Y.device)
    
    # SMOTE samples don't have corresponding PTIDs since they are interpolated
    ptids_aug = None
    
    return X_aug, Y_aug, ptids_aug

def smote_augmentation_full(X, Y, ptids, ptid_count=None):
    """
    SMOTE augmentation - full setting: augment all classes to target PTID count
    Returns only the augmented samples (not including original data)
    
    Args:
        X: Input features [batch_size, feature_dim]
        Y: Labels [batch_size]
        ptids: Patient IDs [batch_size] (not used, kept for compatibility)
        ptid_count: Target count based on unique PTID numbers
    
    Returns:
        X_aug: Only augmented features (original data excluded)
        Y_aug: Only augmented labels (original data excluded)
        ptids_aug: None (SMOTE samples don't have corresponding PTIDs)
    """
    # Convert to numpy for SMOTE
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    # If ptid_count is provided, use it as target
    if ptid_count is not None:
        # Create custom sampling strategy
        unique_classes = np.unique(Y_np)
        sampling_strategy = {cls: ptid_count for cls in unique_classes}
    else:
        # Default SMOTE behavior
        sampling_strategy = 'auto'
    
    # Apply SMOTE only on X features
    smote = SMOTE(random_state=0, sampling_strategy=sampling_strategy)
    X_combined_np, Y_combined_np = smote.fit_resample(X_np, Y_np)
    
    # Extract only the augmented samples (exclude original data)
    original_size = len(X_np)
    X_aug_np = X_combined_np[original_size:]
    Y_aug_np = Y_combined_np[original_size:]
    
    # Convert back to torch tensors
    X_aug = torch.tensor(X_aug_np, dtype=X.dtype, device=X.device)
    Y_aug = torch.tensor(Y_aug_np, dtype=Y.dtype, device=Y.device)
    
    # SMOTE samples don't have corresponding PTIDs since they are interpolated
    ptids_aug = None
    
    return X_aug, Y_aug, ptids_aug

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='adni_ct', help='Type of dataset: adni_ct, adni_amy, adni_fdg, adni_tau')
    parser.add_argument('--device', type=int, default=1, help='GPU device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--method', default='SMOTE', help='Augmentation method')
    
    return parser.parse_args()

def get_data_paths(data_type):
    """
    Get data paths for specified dataset type
    
    Args:
        data_type: One of 'adni_ct', 'adni_amy', 'adni_fdg', 'adni_tau'
    
    Returns:
        Tuple of (data_path, output_dir)
    """
    data_paths = {
        'adni_ct': '/home/user14/bagle/data/ADNI_CT/real.pt',
        'adni_amy': '/home/user14/bagle/data/ADNI_Amy/real.pt',
        'adni_fdg': '/home/user14/bagle/data/ADNI_FDG/real.pt',
        'adni_tau': '/home/user14/bagle/data/ADNI_Tau/real.pt'
    }
    
    output_dirs = {
        'adni_ct': '/home/user14/bagle/data/ADNI_CT/syn',
        'adni_amy': '/home/user14/bagle/data/ADNI_Amy/syn',
        'adni_fdg': '/home/user14/bagle/data/ADNI_FDG/syn',
        'adni_tau': '/home/user14/bagle/data/ADNI_Tau/syn'
    }
    
    if data_type not in data_paths:
        raise ValueError(f"Unknown data type: {data_type}. Must be one of {list(data_paths.keys())}")
    
    return data_paths[data_type], output_dirs[data_type]

def get_ptid_count(data_dict):
    """Get the number of unique PTIDs for full augmentation setting"""
    ptids = data_dict['PTID']
    unique_ptids = len(set(ptids))
    return unique_ptids

def main():
    args = get_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data paths based on data type
    data_path, output_dir = get_data_paths(args.data)
    
    # Load data
    print(f"Loading data from {data_path}")
    data_dict = torch.load(data_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get number of folds
    n_folds = len(torch.unique(data_dict['fold']))
    print(f"Processing {n_folds} folds for {args.data}")
    
    # Get number of unique PTIDs for full augmentation
    ptid_count = get_ptid_count(data_dict)
    print(f"Total unique PTIDs: {ptid_count}")
    
    # Process each fold
    for fold_idx in range(n_folds):
        print(f"\n{'='*50}")
        print(f"Processing Fold {fold_idx}")
        print(f"{'='*50}")
        
        # Get training data for this fold
        X_train, Y_train, PTID_train = create_fold_data(data_dict, fold_idx)
        
        print(f"Fold {fold_idx} - Training samples: {len(X_train)}")
        print(f"Fold {fold_idx} - Feature dimension: {X_train.shape[1]}")
        print(f"Fold {fold_idx} - Original class distribution: {Counter(Y_train.numpy())}")
        
        # Move data to device
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)
        PTID_train = PTID_train.to(device)
        
        # Perform SMOTE augmentation - min setting
        print(f"Performing SMOTE augmentation (min) for fold {fold_idx}...")
        X_min, Y_min, PTID_min = smote_augmentation_min(X_train, Y_train, PTID_train)
        
        # Perform SMOTE augmentation - full setting
        print(f"Performing SMOTE augmentation (full) for fold {fold_idx}...")
        X_full, Y_full, PTID_full = smote_augmentation_full(X_train, Y_train, PTID_train, ptid_count)
        
        # Check class distributions
        original_counter = Counter(Y_train.cpu().numpy())
        min_counter = Counter(Y_min.cpu().numpy())
        full_counter = Counter(Y_full.cpu().numpy())
        
        print(f"Fold {fold_idx} - Original distribution: {original_counter}")
        print(f"Fold {fold_idx} - Min augmented distribution: {min_counter}")
        print(f"Fold {fold_idx} - Full augmented distribution: {full_counter}")
        
        # Save augmented data - minimal augmentation (min setting)
        # Save in the same format as real.pt but only with augmented samples
        # Note: PTID is not saved since SMOTE samples are interpolated and don't correspond to specific patients
        min_data = {
            'X': X_min.unsqueeze(-1),  # Add last dimension to match real.pt format
            'Y': Y_min
        }
        
        # Save augmented data - full augmentation (full setting)
        full_data = {
            'X': X_full.unsqueeze(-1),  # Add last dimension to match real.pt format
            'Y': Y_full
        }
        
        # Save files
        min_filename = f"{args.method}_min_fold{fold_idx}.pt"
        full_filename = f"{args.method}_full_fold{fold_idx}.pt"
        
        min_path = os.path.join(output_dir, min_filename)
        full_path = os.path.join(output_dir, full_filename)
        
        torch.save(min_data, min_path)
        torch.save(full_data, full_path)
        
        print(f"Saved {min_filename} and {full_filename}")
        print(f"Min data shape: {X_min.shape}")
        print(f"Full data shape: {X_full.shape}")

if __name__ == "__main__":
    main()
