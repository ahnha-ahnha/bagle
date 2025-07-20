import torch
import torch.optim as optim

from utils.utility import *
from utils.loader import *
from utils.train import *
from utils.utility import *
from utils.loader import *
from utils.train import *

from models.exact import *
from models.gcn import DDNet
from models.gat import GAT
from models.mlp import MLP, MLP_A
from models.gdc import gdc
from models.agt import *
from models.glas import *
from models.adc import *
from sklearn.svm import SVC

def select_model(args, num_ROI_features, num_used_features, adjacencies, labels):
    if args.model == 'svm':
        model = SVC(kernel='linear')
    elif args.model == 'mlp':
        model = MLP(in_feats = num_ROI_features,
                    hid_feats = args.hidden_units,
                    out_feats = torch.max(labels).item() + 1)
    elif args.model == 'mlp-a':
        # MLP with both features and adjacency matrix
        adj_feats = num_ROI_features * num_ROI_features  # flattened adjacency matrix size
        model = MLP_A(in_feats = num_ROI_features,
                      adj_feats = adj_feats,
                      hid_feats = args.hidden_units,
                      out_feats = torch.max(labels).item() + 1)
    elif args.model == 'agt':
        model = AGT(args,
                    adj_dim=num_ROI_features,
                    in_dim=num_used_features,
                    hid_dim=args.hidden_units,
                    out_dim=torch.max(labels).item() + 1,
                    dropout=args.dropout_rate)
    elif args.model == 'exact':
        if args.layer_num == 1:
            model = Exact1(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 2:
            model = Exact2(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 3:
            model = Exact3(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)
        elif args.layer_num == 4:
            model = Exact4(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)      

    elif args.model == 'gcn' or args.model == 'gdc' or args.model == 'graphheat':
            """
            ajd_dim: # ROI features (edges) 
            in_dim: # used features (nodes)
            hid_dim: # hidden units (weights)
            out_dim: # labels (classes)
            """
            model = DDNet(adj_dim=num_ROI_features,
                        in_dim=num_used_features,
                        hid_dim=args.hidden_units,
                        out_dim=torch.max(labels).item() + 1,
                        dropout=args.dropout_rate)

    elif args.model == 'gat':
        """
        nfeat: # used features (nodes)
        nhid: # hidden units (weights)
        nclass: # labels (classes)
        """
        model = GAT(nfeat=num_used_features,
                    nhid=args.hidden_units,
                    nclass=torch.max(labels).item() + 1,
                    dropout=args.dropout_rate,
                    alpha=args.alpha,
                    adj_sz=adjacencies[0].shape[0],
                    nheads=args.num_head_attentions)
    elif args.model == 'adc':
        """
        nfeat: # used features (nodes)
        nhid: # hidden units (weights)
        nclass: # labels (classes)
        """
        model = ADC(adj_dim=num_ROI_features,
                    in_dim=num_used_features,
                    hid_dim=args.hidden_units,
                    out_dim=torch.max(labels).item() + 1,
                    dropout=args.dropout_rate)
    elif args.model == 'glas':
        model = GLAS1(in_dim=num_used_features,
                      hid_dim=args.hidden_units,
                      out_dim=torch.max(labels).item() + 1,
                      dropout=args.dropout_rate)
    
    return model
        
def select_optimizer(args, model):
    if args.model == 'svm':
        return None  # SVM doesn't need optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, adjacencies, cv_idx=None, pretrained_net=None, pretrained_t=None):
    if args.model == 'svm':
        trainer = SVM_Trainer(args, device, model, data_loader_train, data_loader_test)
    elif args.model == 'mlp':
        trainer = MLP_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test, cv_idx)
    elif args.model == 'mlp-a':
        trainer = MLP_A_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test, cv_idx)
    elif args.model in ['gcn', 'gat', 'gdc', 'adc', 'glas']:
        trainer = GNN_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test, cv_idx)
    elif args.model == 'graphheat':
        trainer = GraphHeat_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test, adjacencies[0].shape[0], cv_idx)
    elif args.model == 'exact':
        trainer = Exact_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, data_loader_test, adjacencies[0].shape[0], cv_idx)
    elif args.model == 'agt': 
        trainer = AGT_Trainer(args, device, model, optimizer, data_loader_train, data_loader_test, adjacencies[0].shape[0], cv_idx, pretrained_net, pretrained_t)
    
    return trainer