import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class MLP(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(self.__class__, self).__init__()
        """
        self.pred = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, out_feats),
        )
        """
        self.pred = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, out_feats),
            nn.LeakyReLU()
        )

    def forward(self, x): # (sample, node, feature)
        x = x.reshape(x.shape[0],-1)
        x = self.pred(x)

        return F.log_softmax(x, dim=1)


class MLP_A(nn.Module):
    def __init__(self, in_feats, adj_feats, hid_feats, out_feats):
        super(self.__class__, self).__init__()
        # Total input features = flattened X features + flattened A features
        total_feats = in_feats + adj_feats
        
        self.pred = nn.Sequential(
            nn.Linear(total_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, out_feats),
            nn.LeakyReLU()
        )

    def forward(self, x, a):  # x: (sample, node, feature), a: (sample, node, node)
        # Flatten both x and a
        x_flat = x.reshape(x.shape[0], -1)  # (sample, node * feature)
        a_flat = a.reshape(a.shape[0], -1)  # (sample, node * node)
        
        # Concatenate flattened x and a
        combined = torch.cat([x_flat, a_flat], dim=1)  # (sample, node * feature + node * node)
        
        output = self.pred(combined)
        return F.log_softmax(output, dim=1)
