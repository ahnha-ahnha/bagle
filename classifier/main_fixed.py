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

def save_summary_excel(data, model, augmentation, results, filename="experiment_summary.xlsx"):
    """
    Save experiment results to Excel file
    
    Args:
        data: Dataset name
        model: Model name  
        augmentation: Augmentation method used
        results: Dictionary with metric results
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
        'Avg_Accuracy': f"{results['accuracy']:.4f} ± {results['accuracy_std']:.4f}",
        'Avg_Precision': f"{results['precision']:.4f} ± {results['precision_std']:.4f}",
        'Avg_Recall': f"{results['recall']:.4f} ± {results['recall_std']:.4f}",
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
    
    # Save to Excel
    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")

def split_by_ptid(ptids, labels, n_splits=5, seed=42):
    """
    Split data by PTID to avoid data leakage
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

### Make argument parser(hyper-parameters)
def get_args():
    parser = argparse.ArgumentParser()
    ### WanDB
    parser.add_argument('--run_name', default='adni_fdg', help='Name of wandb run')
    ### Data
    parser.add_argument('--data', default='adni_fdg', help='Type of dataset: adni_ct, adni_amy, adni_fdg, adni_tau, ppmi')
    parser.add_argument('--nclass', default=5, help='Number of classes')
    parser.add_argument('--adjacency_path', default='/path_to_adj', help='set path to adjacency matrices')
    parser.add_argument('--save_dir', default='./logs/', help='directory for saving weight file')

    ### Condition
    parser.add_argument('--t_init_min', type=float, default=-2.0, help='Init value of t')
    parser.add_argument('--t_init_max', type=float, default=2.0, help='Init value of t')
    parser.add_argument('--seed_num', type=int, default=100, help='Number of random seed')
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

### Main function
def main():
    args = get_args()
    set_randomness(args.seed_num)
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    # Automatically set run_name if not provided
    if args.run_name == 'adni_fdg':  # Default value
        args.run_name = f"{args.data}_{args.model}"
        print(f"Auto-generated run_name: {args.run_name}")

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
        
        # Use sample-level K-fold cross validation for PPMI
        print("Using sample-level K-fold cross validation for PPMI")
        stratified_train_test_split = StratifiedKFold(n_splits=5)
        idx_pairs = []
        for train_idx, test_idx in stratified_train_test_split.split(A, y):
            idx_train = torch.LongTensor(train_idx)
            idx_test = torch.LongTensor(test_idx)
            idx_pairs.append((idx_train, idx_test))
        
    ### Initialize metrics storage
    avl, ava, avac, avpr, avsp, avse, avf1s, aurocs, macro_f1s, macro_aurocs = [[] for _ in range(10)]
    ts = []

    ### Utilize GPUs for computation
    if torch.cuda.is_available() and args.model != 'svm':
        A = A.to(device) 
        X = X.to(device) 
        y = y.to(device) 
        eigenvalues = eigenvalues.to(device) 
        eigenvectors = eigenvectors.to(device) 
    
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

    # Process each fold separately with individual WandB runs
    for i, idx_pair in enumerate(idx_pairs):
        print("\n")
        print(f"=============================== Fold {i+1} ===============================")
        
        # Initialize separate WandB run for each fold
        fold_run_name = f"{args.run_name}_fold_{i+1}"
        wandb.init(project="graph classification", name=fold_run_name, reinit=True)
        wandb.config.update(vars(args))  # Convert args to dict
        wandb.config.update({"fold": i+1})
        
        ### Build data loader
        data_loader_train, data_loader_test = build_data_loader(args, idx_pair, A, X, y, eigenvalues, eigenvectors)

        ### Select the model to use
        model = select_model(args, num_ROI_features, num_used_features, A, y)
        optimizer = select_optimizer(args, model)        
        
        if args.model == 'agt':
            # AGT pretrained model loading logic (simplified)
            p_net, p_t = None, None
            trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, A, cv_idx = i, pretrained_net=p_net, pretrained_t=p_t) 
        elif args.model == 'exact':
            trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, A, cv_idx=i) 
        else: 
            trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, A, cv_idx=i)
            
        ### Train and test
        test_acc_list, test_sens_list, test_prec_list = trainer.train()
        
        ### Get best model results
        model_path = os.path.join(args.save_dir, '{}.pth'.format(i))
        test_model = select_model(args, num_ROI_features, num_used_features, A, y)
        
        # Get all metrics from load_and_test
        test_results = trainer.load_and_test(test_model, model_path)
        if len(test_results) == 11:  # New format with all metrics
            (losses, accuracies, cf_accuracies, cf_precisions, cf_specificities, 
             cf_sensitivities, cf_f1score, auroc, macro_f1, macro_auroc, t) = test_results
        else:  # Legacy format
            if len(test_results) == 8:
                (losses, accuracies, cf_accuracies, cf_precisions, cf_specificities, 
                 cf_sensitivities, cf_f1score, t) = test_results
            else:
                # Handle any other format
                losses = test_results[0] if len(test_results) > 0 else 0.0
                accuracies = test_results[1] if len(test_results) > 1 else 0.0
                cf_accuracies = test_results[2] if len(test_results) > 2 else 0.0
                cf_precisions = test_results[3] if len(test_results) > 3 else 0.0
                cf_specificities = test_results[4] if len(test_results) > 4 else 0.0
                cf_sensitivities = test_results[5] if len(test_results) > 5 else 0.0
                cf_f1score = test_results[6] if len(test_results) > 6 else 0.0
                t = test_results[7] if len(test_results) > 7 else None
            auroc = macro_f1 = macro_auroc = 0.0

        # Store results
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
        
        # Log final test metrics to WandB
        wandb.log({
            "test_loss": losses,
            "test_accuracy": accuracies,
            "test_precision": cf_precisions,
            "test_specificity": cf_specificities,
            "test_sensitivity": cf_sensitivities,
            "test_f1": cf_f1score,
            "test_auroc": auroc,
            "test_macro_f1": macro_f1,
            "test_macro_auroc": macro_auroc,
            "fold": i+1
        })
        
        # Close current fold's WandB run
        wandb.finish()

    class_info = y.tolist()
    cnt = Counter(class_info)

    # Calculate summary statistics
    results = {
        'accuracy': np.mean(ava),
        'accuracy_std': np.std(ava),
        'precision': np.mean(avpr),
        'precision_std': np.std(avpr),
        'recall': np.mean(avse),
        'recall_std': np.std(avse),
        'f1': np.mean(avf1s),
        'f1_std': np.std(avf1s),
        'auroc': np.mean(aurocs),
        'auroc_std': np.std(aurocs),
        'macro_f1': np.mean(macro_f1s),
        'macro_f1_std': np.std(macro_f1s),
        'macro_auroc': np.mean(macro_aurocs),
        'macro_auroc_std': np.std(macro_aurocs)
    }

    ### Show results
    print("--------------- Result ---------------")
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
    print(f"Accuracy:    {results['accuracy']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"Precision:   {results['precision']:.4f} ± {results['precision_std']:.4f}")
    print(f"Recall:      {results['recall']:.4f} ± {results['recall_std']:.4f}")
    print(f"F1:          {results['f1']:.4f} ± {results['f1_std']:.4f}")
    print(f"AUROC:       {results['auroc']:.4f} ± {results['auroc_std']:.4f}")
    print(f"Macro F1:    {results['macro_f1']:.4f} ± {results['macro_f1_std']:.4f}")
    print(f"Macro AUROC: {results['macro_auroc']:.4f} ± {results['macro_auroc_std']:.4f}")

    # Save to Excel summary
    try:
        save_summary_excel(args.data, args.model, "none", results)
    except Exception as e:
        print(f"Error saving to Excel: {e}")

if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    process_time = time.time() - start_time
    hour = int(process_time // 3600)
    minute = int((process_time - hour * 3600) // 60)
    second = int(process_time % 60)
    print(f"\nTime: {hour}:{minute}:{second}")
