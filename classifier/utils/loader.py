import torch
import numpy as np
import pandas as pd
import sys
import os
import random

from torch.linalg import eigh
from os import walk
from .utility import *
from torch.utils.data import TensorDataset, DataLoader

def load_saved_data(args):
    adjacencies = torch.load(os.path.join(args.data_path, 'A.pt'), weights_only=True)
    features = torch.load(os.path.join(args.data_path, 'X.pt'), weights_only=True)
    labels = torch.load(os.path.join(args.data_path, 'labels.pt'), weights_only=True)
    eigenvalues = torch.load(os.path.join(args.data_path, 'eigval.pt'), weights_only=True)
    eigenvectors = torch.load(os.path.join(args.data_path, 'eigvec.pt'), weights_only=True)

    return adjacencies, features, labels, eigenvalues, eigenvectors 

def load_saved_data_with_ptid(args):
    """Load data including PTID information for subject-level splitting"""
    adjacencies = torch.load(os.path.join(args.data_path, 'A.pt'), weights_only=True)
    features = torch.load(os.path.join(args.data_path, 'X.pt'), weights_only=True)
    labels = torch.load(os.path.join(args.data_path, 'labels.pt'), weights_only=True)
    eigenvalues = torch.load(os.path.join(args.data_path, 'eigval.pt'), weights_only=True)
    eigenvectors = torch.load(os.path.join(args.data_path, 'eigvec.pt'), weights_only=True)
    
    # Try to load PTID if available
    ptid_path = os.path.join(args.data_path, 'ptids.pt')
    ptids = None
    if os.path.exists(ptid_path):
        ptids = torch.load(ptid_path, weights_only=True)
        print(f"Loaded PTID information: {len(ptids)} entries")
    else:
        print(f"Warning: PTID file not found at {ptid_path}")
        print("Using sample-level splitting instead of subject-level splitting")

    return adjacencies, features, labels, eigenvalues, eigenvectors, ptids 


def build_data_loader(args, idx_pair, adjacencies, features, labels, eigenvalues, eigenvectors, 
                      X_aug=None, y_aug=None, adj_assignment_method='average', adj_percentile=90):
    idx_train, idx_test = idx_pair

    # Apply GCN-specific processing to real adjacency matrices if needed
    if args.model == 'gcn':
        print("Applying GCN-specific adjacency processing to real data")
        adjacencies = process_adjacency_for_gcn(adjacencies, args.model, normalize=False)

    # Get training data
    train_adjacencies = adjacencies[idx_train]
    train_features = features[idx_train] 
    train_labels = labels[idx_train]
    
    # Add augmentation data if provided
    if X_aug is not None and y_aug is not None:
        print(f"Adding {X_aug.shape[0]} augmented samples to training set")
        
        # Move augmented data to the same device as original data
        X_aug = X_aug.to(features.device)
        y_aug = y_aug.to(labels.device)
        
        # Create adjacency matrices for augmented data
        # Support both Option-R (random) and Option-M (average) assignment methods
        aug_adjacencies = generate_adjacency_for_augmented_data(
            adjacencies[idx_train], labels[idx_train], y_aug, 
            model_type=args.model, assignment_method=adj_assignment_method, 
            percentile=adj_percentile
        )
        aug_adjacencies = aug_adjacencies.to(adjacencies.device)
        
        # Concatenate original training data with augmented data
        train_adjacencies = torch.cat([train_adjacencies, aug_adjacencies], dim=0)
        train_features = torch.cat([train_features, X_aug], dim=0)
        train_labels = torch.cat([train_labels, y_aug], dim=0)
        
        print(f"Final training set size: {train_features.shape[0]} samples")

    if args.model in ['svm', 'mlp', 'mlp-a', 'gcn', 'gat', 'gdc', 'adc', 'glas']:  
        data_train = TensorDataset(train_adjacencies, train_features, train_labels)
        data_test = TensorDataset(adjacencies[idx_test], features[idx_test], labels[idx_test])
    elif args.model in ['graphheat', 'exact', 'agt']:
        # For models that need eigenvalues/eigenvectors, create dummy values for augmented data
        train_eigenvalues = eigenvalues[idx_train]
        train_eigenvectors = eigenvectors[idx_train]
        
        if X_aug is not None and y_aug is not None:
            # Create dummy eigenvalues and eigenvectors for augmented data
            aug_eigenvalues = torch.zeros(X_aug.shape[0], eigenvalues.shape[1])
            aug_eigenvectors = torch.zeros(X_aug.shape[0], eigenvectors.shape[1], eigenvectors.shape[2])
            aug_eigenvalues = aug_eigenvalues.to(eigenvalues.device)
            aug_eigenvectors = aug_eigenvectors.to(eigenvectors.device)
            
            train_eigenvalues = torch.cat([train_eigenvalues, aug_eigenvalues], dim=0)
            train_eigenvectors = torch.cat([train_eigenvectors, aug_eigenvectors], dim=0)
        
        data_train = TensorDataset(train_adjacencies, train_features, train_labels, train_eigenvalues, train_eigenvectors)
        data_test = TensorDataset(adjacencies[idx_test], features[idx_test], labels[idx_test], eigenvalues[idx_test], eigenvectors[idx_test])
    else:
        # Default case for any other models
        print(f"Warning: Unknown model type '{args.model}', using default dataset format")
        data_train = TensorDataset(train_adjacencies, train_features, train_labels)
        data_test = TensorDataset(adjacencies[idx_test], features[idx_test], labels[idx_test])

    data_loader_train = DataLoader(data_train, batch_size=train_features.shape[0], shuffle=True) # Full-batch
    data_loader_test = DataLoader(data_test, batch_size=idx_test.shape[0], shuffle=True) # Full-batch
    
    return data_loader_train, data_loader_test

def build_class_pools(train_adjacencies, train_labels):
    """
    Build class-wise pools of adjacency matrices for synthetic data assignment
    
    Args:
        train_adjacencies: Training adjacency matrices [n_train, n_roi, n_roi]
        train_labels: Training labels [n_train]
    
    Returns:
        class_pools: Dictionary {class_idx: [adj_matrices]}
    """
    class_pools = {}
    unique_classes = torch.unique(train_labels)
    
    for class_idx in unique_classes:
        class_mask = (train_labels == class_idx)
        class_adjacencies = train_adjacencies[class_mask]
        class_pools[class_idx.item()] = class_adjacencies
    
    return class_pools

def assign_random_adjacency(aug_labels, class_pools):
    """
    Option-R: Assign random adjacency matrices from same class pool
    
    Args:
        aug_labels: Augmented data labels [n_aug]
        class_pools: Dictionary {class_idx: [adj_matrices]}
    
    Returns:
        aug_adjacencies: Adjacency matrices for augmented data [n_aug, n_roi, n_roi]
    """
    n_aug = aug_labels.shape[0]
    device = aug_labels.device
    
    # Get matrix dimensions from first available class
    first_class_matrices = next(iter(class_pools.values()))
    n_roi = first_class_matrices.shape[1]
    
    aug_adjacencies = torch.zeros(n_aug, n_roi, n_roi, device=device)
    
    print(f"Assigning random adjacency matrices for {n_aug} synthetic samples (Option-R)")
    
    class_assignment_count = {}
    for i, label in enumerate(aug_labels):
        label_idx = label.item()
        
        if label_idx in class_pools and len(class_pools[label_idx]) > 0:
            # Randomly select from same class pool
            random_idx = torch.randint(0, len(class_pools[label_idx]), (1,)).item()
            aug_adjacencies[i] = class_pools[label_idx][random_idx]
            class_assignment_count[label_idx] = class_assignment_count.get(label_idx, 0) + 1
        else:
            # Fallback: use identity matrix
            aug_adjacencies[i] = torch.eye(n_roi, device=device)
            print(f"Warning: No adjacency matrices for class {label_idx}, using identity matrix")
    
    print("Random adjacency assignment:")
    for class_idx, count in class_assignment_count.items():
        print(f"  Class {class_idx}: {count} synthetic samples")
    
    return aug_adjacencies

def make_class_avg_normalized(class_pools, percentile=90):
    """
    Option-M: Create class-wise average adjacency matrices with sparsification and normalization
    
    Args:
        class_pools: Dictionary {class_idx: [adj_matrices]}
        percentile: Percentile threshold for sparsification (default 90)
    
    Returns:
        class_avg_adjacencies: Dictionary {class_idx: normalized_avg_adjacency}
    """
    class_avg_adjacencies = {}
    
    print(f"Creating class-wise average adjacency matrices with {percentile}% sparsification")
    
    for class_idx, adj_matrices in class_pools.items():
        if len(adj_matrices) == 0:
            continue
            
        n_roi = adj_matrices.shape[1]
        device = adj_matrices.device
        
        # Convert to numpy for processing
        adj_matrices_np = adj_matrices.cpu().numpy()
        
        # 1) Calculate class average
        A_bar = np.mean(adj_matrices_np, axis=0)
        A_bar = 0.5 * (A_bar + A_bar.T)  # Ensure symmetry
        np.fill_diagonal(A_bar, 0)  # Remove diagonal for degree calculation
        
        # 2) Calculate original average degree to maintain target edge count
        d_orig = np.mean([A.sum(1).mean() for A in adj_matrices_np])
        E_target = int(round(n_roi * d_orig / 2))
        
        # 3) Apply percentile-based sparsification
        if E_target > 0:
            upper_tri_indices = np.triu_indices(n_roi, k=1)
            upper_tri_values = A_bar[upper_tri_indices]
            
            if len(upper_tri_values) > 0:
                # Use percentile threshold
                threshold = np.percentile(upper_tri_values, percentile)
                A_thr = np.where(A_bar >= threshold, A_bar, 0)
            else:
                A_thr = A_bar.copy()
        else:
            A_thr = np.zeros_like(A_bar)
        
        # Ensure symmetry after thresholding
        A_thr = np.maximum(A_thr, A_thr.T)
        
        # 4) Add self-loops and apply normalization for GCN (normalize=False)
        np.fill_diagonal(A_thr, 1)
        
        # Calculate degree and apply D^(-1/2) * A * D^(-1/2) normalization
        degree = A_thr.sum(1)
        degree[degree == 0] = 1  # Avoid division by zero
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        A_normalized = D_inv_sqrt @ A_thr @ D_inv_sqrt
        
        # Convert back to torch tensor
        class_avg_adjacencies[class_idx] = torch.tensor(A_normalized, 
                                                       dtype=adj_matrices.dtype, 
                                                       device=device)
        
        print(f"Class {class_idx}: processed {len(adj_matrices)} matrices, "
              f"original avg degree: {d_orig:.2f}, target edges: {E_target}")
    
    return class_avg_adjacencies

def assign_average_adjacency(aug_labels, class_avg_adjacencies):
    """
    Option-M: Assign pre-computed class average adjacency matrices
    
    Args:
        aug_labels: Augmented data labels [n_aug]
        class_avg_adjacencies: Dictionary {class_idx: normalized_avg_adjacency}
    
    Returns:
        aug_adjacencies: Adjacency matrices for augmented data [n_aug, n_roi, n_roi]
    """
    n_aug = aug_labels.shape[0]
    device = aug_labels.device
    
    # Get matrix dimensions from first available class
    first_class_adj = next(iter(class_avg_adjacencies.values()))
    n_roi = first_class_adj.shape[0]
    
    aug_adjacencies = torch.zeros(n_aug, n_roi, n_roi, device=device)
    
    print(f"Assigning class average adjacency matrices for {n_aug} synthetic samples (Option-M)")
    
    class_assignment_count = {}
    for i, label in enumerate(aug_labels):
        label_idx = label.item()
        
        if label_idx in class_avg_adjacencies:
            aug_adjacencies[i] = class_avg_adjacencies[label_idx]
            class_assignment_count[label_idx] = class_assignment_count.get(label_idx, 0) + 1
        else:
            # Fallback: use identity matrix
            aug_adjacencies[i] = torch.eye(n_roi, device=device)
            print(f"Warning: No average adjacency for class {label_idx}, using identity matrix")
    
    print("Average adjacency assignment:")
    for class_idx, count in class_assignment_count.items():
        print(f"  Class {class_idx}: {count} synthetic samples")
    
    return aug_adjacencies

def generate_adjacency_for_augmented_data(train_adjacencies, train_labels, aug_labels, 
                                        model_type=None, assignment_method='average', 
                                        percentile=90):
    """
    Generate adjacency matrices for augmented data using two assignment methods
    
    Args:
        train_adjacencies: Training adjacency matrices [n_train, n_roi, n_roi]
        train_labels: Training labels [n_train]
        aug_labels: Augmented data labels [n_aug]
        model_type: Type of model ('gcn', 'gat', etc.) for specific processing
        assignment_method: 'random' (Option-R) or 'average' (Option-M)
        percentile: Percentile threshold for sparsification (only for Option-M)
    
    Returns:
        aug_adjacencies: Adjacency matrices for augmented data [n_aug, n_roi, n_roi]
    """
    n_aug = aug_labels.shape[0]
    print(f"Generating adjacency matrices for {n_aug} synthetic samples using {assignment_method} method")
    
    # Build class-wise pools
    class_pools = build_class_pools(train_adjacencies, train_labels)
    
    if assignment_method == 'random':
        # Option-R: Random assignment from same class
        aug_adjacencies = assign_random_adjacency(aug_labels, class_pools)
    elif assignment_method == 'average':
        # Option-M: Class average with sparsification and normalization
        class_avg_adjacencies = make_class_avg_normalized(class_pools, percentile=percentile)
        aug_adjacencies = assign_average_adjacency(aug_labels, class_avg_adjacencies)
    else:
        raise ValueError(f"Unknown assignment method: {assignment_method}. Use 'random' or 'average'")
    
    # Apply additional processing for specific model types if needed
    if model_type == 'gcn' and assignment_method == 'random':
        # For random assignment with GCN, we might want additional processing
        print("Applying GCN-specific adjacency processing for random assignment")
        aug_adjacencies = process_adjacency_for_gcn(aug_adjacencies, model_type, normalize=False)
    
    return aug_adjacencies

def apply_percentile_sparsification_and_normalization(adjacency_matrix, percentile=95):
    """
    Apply percentile-based sparsification and normalization to adjacency matrix.
    This is specifically for GCN models with normalize=False.
    
    Steps:
    1. Calculate percentile threshold
    2. Sparsify using threshold
    3. Restore self-loops
    4. Apply symmetric normalization
    5. Ensure average degree matches original
    
    Args:
        adjacency_matrix: Input adjacency matrix [n_roi, n_roi]
        percentile: Percentile threshold for sparsification (default 95)
    
    Returns:
        Processed adjacency matrix with sparsification and normalization
    """
    # Work with a copy to avoid modifying original
    adj = adjacency_matrix.clone()
    n_roi = adj.shape[0]
    
    # Calculate original average degree (excluding self-loops)
    adj_no_diag = adj.clone()
    adj_no_diag.fill_diagonal_(0)
    original_avg_degree = torch.sum(adj_no_diag > 0).float() / n_roi
    
    # Step 1: Calculate percentile threshold from upper triangular part (excluding diagonal)
    upper_tri_mask = torch.triu(torch.ones_like(adj), diagonal=1).bool()
    upper_tri_values = adj[upper_tri_mask]
    threshold = torch.quantile(upper_tri_values, percentile / 100.0)
    
    # Step 2: Sparsify using threshold
    adj_sparse = torch.where(adj >= threshold, adj, torch.zeros_like(adj))
    
    # Step 3: Restore self-loops (diagonal elements)
    adj_sparse.fill_diagonal_(1.0)
    
    # Step 4: Apply symmetric normalization
    # Calculate degree matrix
    degree = torch.sum(adj_sparse, dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    
    # Create degree matrix
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    
    # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
    adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj_sparse), D_inv_sqrt)
    
    # Step 5: Ensure average degree matches original (approximately)
    # Calculate new average degree (excluding self-loops)
    adj_normalized_no_diag = adj_normalized.clone()
    adj_normalized_no_diag.fill_diagonal_(0)
    new_avg_degree = torch.sum(adj_normalized_no_diag > 0).float() / n_roi
    
    # Optional: Print statistics for debugging
    if torch.isnan(adj_normalized).any():
        print(f"Warning: NaN values detected in normalized adjacency matrix")
        # Replace NaN with zeros and restore diagonal
        adj_normalized = torch.where(torch.isnan(adj_normalized), torch.zeros_like(adj_normalized), adj_normalized)
        adj_normalized.fill_diagonal_(1.0)
    
    return adj_normalized

def process_adjacency_for_gcn(adjacency_matrices, model_type, normalize=True):
    """
    Process adjacency matrices for GCN models.
    
    Args:
        adjacency_matrices: Input adjacency matrices [n_samples, n_roi, n_roi]
        model_type: Type of model ('gcn', 'gat', etc.)
        normalize: Whether to apply normalization (False for GCN with normalize=False)
    
    Returns:
        Processed adjacency matrices
    """
    if model_type == 'gcn' and not normalize:
        print("Applying percentile-based sparsification and normalization for GCN (normalize=False)")
        processed_adj = torch.zeros_like(adjacency_matrices)
        
        for i in range(adjacency_matrices.shape[0]):
            processed_adj[i] = apply_percentile_sparsification_and_normalization(adjacency_matrices[i])
        
        return processed_adj
    else:
        # For other models or GCN with normalize=True, return as-is
        return adjacency_matrices
