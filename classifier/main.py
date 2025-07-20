import numpy as np
import random
import torch
import argparse
import time
import pickle
import pandas as pd

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from utils.train import *
from utils.loader import *
from utils.utility import *
from utils.model import *
from datetime import datetime

import wandb

def split_by_ptid(ptids, labels, n_splits=5, seed=0):
    """
    Split data by PTID to av            if syn_data is not None:
                X_aug, y_aug = syn_data
                # Move augmented data to the same device as original data
                X_aug = X_aug.to(device)
                y_aug = y_aug.to(device)
                print(f"Using {augmentation_name} augmentation: {X_aug.shape[0]} additional samples")
        
        ### Build data loader
        data_loader_train, data_loader_test = build_data_loader(args, idx_pair, A, X, y, eigenvalues, eigenvectors, X_aug, y_aug, args.adj_assignment, args.adj_percentile)
    
    Args:
        ptids: List or array of patient IDs
        labels: List or array of labels corresponding to each sample
        n_splits: Number of folds for cross-validation
        seed: Random seed for reproducibility
    
    Returns:
        List of (train_idx, test_idx) tuples for each fold
    """
    # Convert to numpy arrays if they aren't already
    ptids = np.array(ptids)
    labels = np.array(labels)
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({"PTID": ptids, "label": labels, "idx": range(len(ptids))})
    
    # Get representative label for each subject (most frequent label)
    subj_label = df.groupby("PTID")["label"].agg(lambda x: x.value_counts().index[0])
    
    subjects = subj_label.index.values
    sub_labels = subj_label.values
    
    # Stratified K-fold on subjects
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_splits = []
    
    for fold, (train_subj_idx, test_subj_idx) in enumerate(skf.split(subjects, sub_labels)):
        train_subjects = subjects[train_subj_idx]
        test_subjects = subjects[test_subj_idx]
        
        # Get all sample indices for train and test subjects
        train_idx = df[df["PTID"].isin(train_subjects)]["idx"].values
        test_idx = df[df["PTID"].isin(test_subjects)]["idx"].values
        
        fold_splits.append((train_idx, test_idx))
        
        print(f"Fold {fold}: {len(train_subjects)} train subjects ({len(train_idx)} samples), "
              f"{len(test_subjects)} test subjects ({len(test_idx)} samples)")
    
    return fold_splits

def load_real_pt_data(data_type):
    """
    Load real.pt data for specified ADNI dataset type
    
    Args:
        data_type: One of 'adni_ct', 'adni_amy', 'adni_fdg', 'adni_tau'
    
    Returns:
        Tuple of (A, X, y, eigenvalues, eigenvectors, ptids, ages, sex, folds)
    """
    data_paths = {
        'adni_ct': '/home/user14/bagle/data/ADNI_CT/real.pt',
        'adni_amy': '/home/user14/bagle/data/ADNI_Amy/real.pt',
        'adni_fdg': '/home/user14/bagle/data/ADNI_FDG/real.pt',
        'adni_tau': '/home/user14/bagle/data/ADNI_Tau/real.pt'
    }
    
    if data_type not in data_paths:
        raise ValueError(f"Unknown data type: {data_type}. Must be one of {list(data_paths.keys())}")
    
    data_path = data_paths[data_type]
    print(f"Loading data from: {data_path}")
    
    # Load the real.pt file
    data = torch.load(data_path)
    
    # Extract all components
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
    """
    Create train/test splits based on existing fold information
    
    Args:
        folds: Tensor or array containing fold assignments for each sample
        n_folds: Number of folds (default: 5)
    
    Returns:
        List of (train_idx, test_idx) tuples for each fold
    """
    if isinstance(folds, torch.Tensor):
        folds = folds.numpy()
    
    fold_splits = []
    
    for fold_num in range(n_folds):
        # Test indices: samples in current fold
        test_idx = np.where(folds == fold_num)[0]
        # Train indices: samples in all other folds
        train_idx = np.where(folds != fold_num)[0]
        
        fold_splits.append((train_idx, test_idx))
        
        print(f"Fold {fold_num}: {len(train_idx)} train samples, {len(test_idx)} test samples")
    
    return fold_splits

def load_syn_data(data_type, augmentation, fold_idx):
    """
    Load synthetic augmented data for specified fold
    
    Args:
        data_type: One of 'adni_ct', 'adni_amy', 'adni_fdg', 'adni_tau'
        augmentation: Augmentation method ('SMOTE_min', 'SMOTE_full')
        fold_idx: Fold index (0-4)
    
    Returns:
        Tuple of (X_syn, y_syn) or None if file doesn't exist
    """
    syn_dirs = {
        'adni_ct': '/home/user14/bagle/data/ADNI_CT/syn',
        'adni_amy': '/home/user14/bagle/data/ADNI_Amy/syn',
        'adni_fdg': '/home/user14/bagle/data/ADNI_FDG/syn',
        'adni_tau': '/home/user14/bagle/data/ADNI_Tau/syn'
    }
    
    if data_type not in syn_dirs:
        return None
        
    syn_dir = syn_dirs[data_type]
    
    # Extract method from augmentation (e.g., 'SMOTE_min' -> 'SMOTE', 'min')
    if '_' in augmentation:
        method, setting = augmentation.split('_', 1)
        filename = f"{method}_{setting}_fold{fold_idx}.pt"
    else:
        filename = f"{augmentation}_fold{fold_idx}.pt"
    
    syn_path = os.path.join(syn_dir, filename)
    
    if not os.path.exists(syn_path):
        print(f"Warning: Synthetic data file not found: {syn_path}")
        return None
    
    try:
        syn_data = torch.load(syn_path)
        X_syn = syn_data['X']  # [n_samples, n_features, 1]
        y_syn = syn_data['Y']  # [n_samples]
        
        print(f"Loaded synthetic data from {filename}: {X_syn.shape[0]} samples")
        print(f"SMOTE synthetic data - using class-based adjacency matrices")
        
        return X_syn, y_syn
    except Exception as e:
        print(f"Error loading synthetic data from {syn_path}: {e}")
        return None

### Make argument parser(hyper-parameters)
def get_args():
    parser = argparse.ArgumentParser()
    ### Data
    parser.add_argument('--data', default='adni_fdg', help='Type of dataset: adni_ct, adni_amy, adni_fdg, adni_tau, ppmi')
    parser.add_argument('--nclass', default=5, help='Number of classes')
    parser.add_argument('--adjacency_path', default='/path_to_adj', help='set path to adjacency matrices')
    parser.add_argument('--save_dir', default='./logs/', help='directory for saving weight file')
    
    ### Augmentation
    parser.add_argument('--augmentation', default='NoAug', help='Augmentation method: NoAug, SMOTE')
    parser.add_argument('--aug_level', default=None, help='Augmentation level: min, full (only used when augmentation=SMOTE)')
    parser.add_argument('--adj_assignment', default='average', help='Adjacency assignment method for synthetic data: random, average')
    parser.add_argument('--adj_percentile', type=float, default=90, help='Percentile threshold for adjacency sparsification (only for average method)')

    ### Condition
    parser.add_argument('--t_init_min', type=float, default=-2.0, help='Init value of t')
    parser.add_argument('--t_init_max', type=float, default=2.0, help='Init value of t')
    parser.add_argument('--seed_num', type=int, default=0, help='Number of random seed')
    parser.add_argument('--device', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--model', type=str, default='svm', help='Models to use') 
    parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')

    ### Experiment
    parser.add_argument('--beta', type=float, default=0.005, help='weight of temporal regularization, alpha in the paper') 
    parser.add_argument('--batch_size', type=int, default=512, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--hidden_units', type=int, default=8, help='Number of hidden units')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learing rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer adam/sgd')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 regularization') # 0.0005

    ### Parameters for training GAT 
    parser.add_argument('--num_head_attentions', type=int, default=16, help='Number of head attentions')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for the leaky relu')

    ### Parameters for training Exact 
    parser.add_argument('--use_t_local', type=int, default=1, help='Whether t is local or global (0:global / 1:local)')
    parser.add_argument('--t_lr', type=float, default=1, help='t learning rate')
    parser.add_argument('--t_loss_threshold', type=float, default=0.01, help='t loss threshold')
    parser.add_argument('--t_lambda', type=float, default=1, help='t lambda of loss function')
    parser.add_argument('--t_threshold', type=float, default=0.1, help='t threshold')
    
    args = parser.parse_args()
    return args

### Control the randomness of all experiments
def set_randomness(seed_num):
    torch.manual_seed(seed_num) # Pytorch randomness
    np.random.seed(seed_num) # Numpy randomness
    random.seed(seed_num) # Python randomness
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num) # Current GPU randomness
        torch.cuda.manual_seed_all(seed_num) # Multi GPU randomness

### Save experiment results to Excel summary file
def save_summary_excel(data, model, augmentation, results, adj_assignment=None, filename="experiment_summary.xlsx"):
    """
    Save experiment results to Excel file
    
    Args:
        data: Dataset name
        model: Model name  
        augmentation: Augmentation method used
        results: Dictionary with metric results
        adj_assignment: Adjacency assignment method (random/average)
        filename: Excel filename
    """
    import pandas as pd
    import os
    from datetime import datetime
    
    # Create results row
    row = {
        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Data': data,
        'Model': model,
        'Augmentation': augmentation,
        'Adj_Assignment': adj_assignment if adj_assignment else 'N/A',
        'Avg_Accuracy': f"{results['acc']:.4f} ± {results['acc_std']:.4f}",
        'Avg_Precision': f"{results['prec']:.4f} ± {results['prec_std']:.4f}",
        'Avg_Recall': f"{results['rec']:.4f} ± {results['rec_std']:.4f}",
        'Avg_F1': f"{results['f1']:.4f} ± {results['f1_std']:.4f}",
        'Avg_AUROC': f"{results['auroc']:.4f} ± {results['auroc_std']:.4f}",
        'Avg_Macro_F1': f"{results['macro_f1']:.4f} ± {results['macro_f1_std']:.4f}",
        'Avg_Macro_AUROC': f"{results['macro_auroc']:.4f} ± {results['macro_auroc_std']:.4f}",
        'Notes': ''
    }
    
    # Load existing data or create new DataFrame
    if os.path.exists(filename):
        try:
            df = pd.read_excel(filename)
        except:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    # Add new row
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    # Define column order
    desired_columns = [
        'Date', 'Data', 'Model', 'Augmentation', 'Adj_Assignment',
        'Avg_Accuracy', 'Avg_Precision', 'Avg_Recall', 'Avg_F1', 
        'Avg_AUROC', 'Avg_Macro_F1', 'Avg_Macro_AUROC', 'Notes'
    ]
    
    # Remove any old columns that shouldn't be there
    columns_to_remove = ['random', 'average']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Reorder columns according to desired order
    existing_columns = [col for col in desired_columns if col in df.columns]
    df = df[existing_columns]
    
    # Save to Excel
    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")

def generate_run_name(args):
    """
    Generate run name automatically based on arguments
    Format: {data}_{model}_{augmentation}_{level}_{adj_method}
    """
    if args.augmentation == 'NoAug':
        run_name = f"{args.data}_{args.model}_NoAug"
    elif args.augmentation == 'SMOTE':
        # Include adjacency assignment method for augmented data
        adj_suffix = f"_{args.adj_assignment}" if args.adj_assignment != 'average' else ""
        run_name = f"{args.data}_{args.model}_SMOTE_{args.aug_level}{adj_suffix}"
    else:
        # For other augmentation methods, include adjacency assignment method
        adj_suffix = f"_{args.adj_assignment}" if args.adj_assignment != 'average' else ""
        run_name = f"{args.data}_{args.model}_{args.augmentation}{adj_suffix}"
    
    return run_name

### Main function
def main():
    args = get_args()
    set_randomness(args.seed_num)
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    # Error handling: Check if NoAug is selected but aug_level is provided
    if args.augmentation == 'NoAug' and args.aug_level is not None:
        print("Error: aug_level should not be specified when augmentation is NoAug")
        print("Please use --augmentation SMOTE with --aug_level min/full, or use --augmentation NoAug without --aug_level")
        exit(1)

    # Automatically generate run_name
    run_name = generate_run_name(args)
    print(f"Auto-generated run_name: {run_name}")

    # Initialize summary results for Excel file
    summary_results = []
    
    # Load data based on dataset type
    if args.data in ['adni_ct', 'adni_amy', 'adni_fdg', 'adni_tau']:
        # Load real.pt data
        A, X, y, eigenvalues, eigenvectors, ptids, ages, sex, folds = load_real_pt_data(args.data)
        args.nclass = len(torch.unique(y))
        print(f"Number of classes: {args.nclass}")
        
        # Create fold splits from existing fold information
        print("Using existing fold information for K-fold cross validation")
        fold_splits = create_fold_splits_from_existing(folds, n_folds=5)
        
        # Convert to torch tensors for idx_pairs
        idx_pairs = []
        for train_idx, test_idx in fold_splits:
            idx_train = torch.LongTensor(train_idx)
            idx_test = torch.LongTensor(test_idx)
            idx_pairs.append((idx_train, idx_test))
            
    elif args.data == 'adni_ct' or args.data == 'adni_fdg':
        ### Load fully-preprocessed data (legacy support)
        if args.data == 'adni_ct':
            args.data_path = '/home/user23/AGT/data/pt/CT_0'
            used_features = ['cortical thickness']
        elif args.data == 'adni_fdg':
            args.data_path = '/home/user23/AGT/data/pt/FDG'
            used_features = ['FDG SUVR']

        # Load data with PTID information for subject-level splitting
        A, X, y, eigenvalues, eigenvectors, ptids = load_saved_data_with_ptid(args)
        ages, sex, folds = None, None, None
        args.nclass = 5

    elif args.data == 'ppmi':
        with open('data_path_to_ppmi', 'rb') as fr:
            data = pickle.load(fr)
            A = data[0]
            X = data[1]
            y = data[2]
            eigenvectors = data[3]
            eigenvalues = data[4]
        ptids, ages, sex, folds = None, None, None, None
        args.nclass = 3
        
    ### K-fold cross validation
    avl, ava, avac, avpr, avsp, avse, avf1s, aurocs, macro_f1s, macro_aurocs = [[] for _ in range(10)]
    average_acc_per_fold = []
    average_sens_per_fold = []
    average_prec_per_fold = []
    ts = []

    # Handle different data loading scenarios
    if args.data in ['adni_ct', 'adni_amy', 'adni_fdg', 'adni_tau']:
        # idx_pairs already created above from fold information
        pass
    elif args.data in ['adni_ct', 'adni_fdg'] and ptids is not None:
        # Use PTID-based splitting if PTID information is available
        print("Using subject-level (PTID-based) K-fold cross validation to avoid data leakage")
        fold_splits = split_by_ptid(ptids, y.cpu().numpy() if isinstance(y, torch.Tensor) else y, 
                                   n_splits=5, seed=args.seed_num)
        idx_pairs = []
        for train_idx, test_idx in fold_splits:
            idx_train = torch.LongTensor(train_idx)
            idx_test = torch.LongTensor(test_idx)
            idx_pairs.append((idx_train, idx_test))
    else:
        print("Using sample-level K-fold cross validation (Warning: potential data leakage)")
        stratified_train_test_split = StratifiedKFold(n_splits=5)
        idx_pairs = []
        for train_idx, test_idx in stratified_train_test_split.split(A, y):
            idx_train = torch.LongTensor(train_idx)
            idx_test = torch.LongTensor(test_idx)
            idx_pairs.append((idx_train, idx_test))

    ### Utilize GPUs for computation
    if torch.cuda.is_available() and args.model != 'svm':
        A = A.to(device) # Shape: (# subjects, # ROI feature, # ROI X)
        X = X.to(device) # Shape: (# subjects, # ROI X, # used X)
        y = y.to(device) # Shape: (# subjects)
    
        eigenvalues = eigenvalues.to(device) # Shape: (# subjects, # ROI feature)
        eigenvectors = eigenvectors.to(device) # Shape: (# subject, # ROI_feature, # ROI_feature)
        # laplacians = laplacians.to(device)
    
    if args.model == 'agt':
        num_ROI_features = X.shape[1]
        num_used_features = X.shape[2] * 2 # stack[x, xs]
    elif args.model == 'svm':
        num_ROI_features = None
        num_used_features = None
    else:
        num_ROI_features = X.shape[1]
        num_used_features = X.shape[2] 


    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir_ = os.path.join(args.save_dir, current_time)
    args.save_dir = save_dir_
    os.makedirs(save_dir_, exist_ok=False)
    print('save directory: ', args.save_dir )

    # Initialize lists to store results for all folds
    all_fold_results = []

    for i, idx_pair in enumerate(idx_pairs):
        print("\n")
        print(f"=============================== Fold {i+1} ===============================")
        
        # Load augmentation data if specified
        X_aug, y_aug = None, None
        if args.augmentation == 'SMOTE' and args.aug_level in ['min', 'full']:
            augmentation_name = f'SMOTE_{args.aug_level}'
            syn_data = load_syn_data(args.data, augmentation_name, i)
            if syn_data is not None:
                X_aug, y_aug = syn_data
                # Move augmented data to the same device as original data
                X_aug = X_aug.to(device)
                y_aug = y_aug.to(device)
                print(f"Using {augmentation_name} augmentation: {X_aug.shape[0]} additional samples")
        
        ### Build data loader
        data_loader_train, data_loader_test = build_data_loader(args, idx_pair, A, X, y, eigenvalues, eigenvectors, X_aug, y_aug, args.adj_assignment, args.adj_percentile)

        ### Select the model to use
        model = select_model(args, num_ROI_features, num_used_features, A, y)

        optimizer = select_optimizer(args, model)        
        if args.model == 'agt':
            if args.data == 'adni_ct':
                data_pretrain_path = 'trained_exact/adni_ct'
                if i == 0:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_0_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_0_t.pt'))
                elif i == 1:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_1_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_1_t.pt'))
                elif i == 2:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_2_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_2_t.pt'))
                elif i == 3:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_3_model.pt')) 
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_3_t.pt')) 
                elif i == 4:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_4_model.pt'))  
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_4_t.pt'))
            
            elif args.data == 'ppmi':
                data_pretrain_path = 'trained_exact/ppmi'
                if i == 0:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_0_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_0_t.pt'))
                elif i == 1:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_1_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_1_t.pt'))
                elif i == 2:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_2_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_2_t.pt'))
                elif i == 3:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_3_model.pt')) 
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_3_t.pt')) 
                elif i == 4:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_4_model.pt'))  
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_4_t.pt'))

            elif args.data == 'adni_fdg':
                data_pretrain_path = 'trained_exact/adni_fdg'
                if i == 0:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_0_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_0_t.pt'))
                elif i == 1:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_1_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_1_t.pt'))
                elif i == 2:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_2_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_2_t.pt'))
                elif i == 3:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_3_model.pt')) 
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_3_t.pt')) 
                elif i == 4:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_4_model.pt'))  
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_4_t.pt'))

            trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, A, cv_idx = i, pretrained_net=p_net, pretrained_t=p_t) 

        elif args.model == 'exact':
            trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, A, cv_idx=i) 
        else: 
            trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, A, cv_idx=i)
        
        # Initialize WandB for this fold
        wandb.init(
            project=f"{args.data}_{args.model}",
            name=f"{run_name}_fold_{i+1}",
            config=vars(args),
            reinit=True
        )
            
        ### Train and test
        val_acc_list, val_sens_list, val_prec_list = trainer.train()
        
        average_acc_per_fold.append(val_acc_list)
        average_sens_per_fold.append(val_sens_list)
        average_prec_per_fold.append(val_prec_list)

        ### best model 
        model_path = os.path.join(args.save_dir, '{}.pth'.format(i)) # get model from logs/datetime/
        test_model = select_model(args, num_ROI_features, num_used_features, A, y)
        losses, accuracies, cf_accuracies, cf_precisions, cf_specificities, cf_sensitivities, cf_f1score, auroc, macro_f1, macro_auroc, t = trainer.load_and_test(test_model, model_path)

        avl.append(losses)
        ava.append(accuracies)
        avac.append(cf_accuracies)
        avpr.append(cf_precisions)
        avsp.append(cf_specificities)
        avse.append(cf_sensitivities)
        avf1s.append(cf_f1score)
        aurocs.append(auroc)
        macro_f1s.append(macro_f1)
        macro_aurocs.append(macro_auroc)
        ts.append(t)

        # Log final test metrics for this fold
        fold_results = {
            'test_acc': round(accuracies, 4),
            'test_prec': round(cf_precisions, 4),
            'test_rec': round(cf_sensitivities, 4),
            'test_f1': round(cf_f1score, 4),
            'test_auroc': round(auroc, 4),
            'test_macro_f1': round(macro_f1, 4),
            'test_macro_auroc': round(macro_auroc, 4)
        }
        
        wandb.log(fold_results)
        all_fold_results.append(fold_results)
        
        # Finish this fold's WandB run
        wandb.finish()
        
        # Clean up GPU memory after each fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory cleared after fold {i + 1}")

    class_info = y.tolist()
    cnt = Counter(class_info)

    # Remove all the old wandb.config.update calls and validation logic
    # since we now handle WandB per fold
    
    ### Show results
    print("--------------- Result ---------------")
    if args.data == 'adni':
        print(f"Used X:        {used_features}")
    print(f"Label distribution:   {cnt}")
    print(f"5-Fold test loss:     {avl}")
    print(f"5-Fold test accuracy: {ava}")
    print("---------- Confusion Matrix ----------")
    print(f"5-Fold precision:     {avpr}")
    print(f"5-Fold specificity:   {avsp}")
    print(f"5-Fold sensitivity:   {avse}")
    print(f"5-Fold f1 score:      {avf1s}")
    print(f"5-Fold AUROC:         {aurocs}")
    print(f"5-Fold Macro F1:      {macro_f1s}")
    print(f"5-Fold Macro AUROC:   {macro_aurocs}")
    print("-------------- Mean, Std --------------")
    print(f"Acc:   {np.mean(ava):.4f} ± {np.std(ava):.4f}")
    print(f"Prec:  {np.mean(avpr):.4f} ± {np.std(avpr):.4f}")
    print(f"Rec:   {np.mean(avse):.4f} ± {np.std(avse):.4f}")
    print(f"F1:    {np.mean(avf1s):.4f} ± {np.std(avf1s):.4f}")
    print(f"AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"Macro F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(f"Macro AUROC: {np.mean(macro_aurocs):.4f} ± {np.std(macro_aurocs):.4f}")
    
    # Save Excel summary to central summary directory
    try:
        summary_results = {
            'acc': np.mean(ava),
            'acc_std': np.std(ava),
            'prec': np.mean(avpr),
            'prec_std': np.std(avpr),
            'rec': np.mean(avse),
            'rec_std': np.std(avse),
            'f1': np.mean(avf1s),
            'f1_std': np.std(avf1s),
            'auroc': np.mean(aurocs),
            'auroc_std': np.std(aurocs),
            'macro_f1': np.mean(macro_f1s),
            'macro_f1_std': np.std(macro_f1s),
            'macro_auroc': np.mean(macro_aurocs),
            'macro_auroc_std': np.std(macro_aurocs)
        }
        
        # Save to central summary directory
        summary_dir = "/home/user14/bagle/summary"
        os.makedirs(summary_dir, exist_ok=True)
        excel_path = os.path.join(summary_dir, "experiment_summary.xlsx")
        
        save_summary_excel(
            data=args.data,
            model=args.model,
            augmentation=f'SMOTE_{args.aug_level}' if args.aug_level != 'NoAug' else 'NoAug',
            results=summary_results,
            adj_assignment=args.adj_assignment if hasattr(args, 'adj_assignment') else None,
            filename=excel_path
        )
        print(f"Excel summary saved to: {excel_path}")
        
    except Exception as e:
        print(f"Error saving Excel summary: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up GPU memory after experiment completion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared after experiment completion")


if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    process_time = time.time() - start_time
    hour = int(process_time // 3600)
    minute = int((process_time - hour * 3600) // 60)
    second = int(process_time % 60)
    print(f"\nTime: {hour}:{minute}:{second}")
    
    # Final GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Final GPU memory cleanup completed")