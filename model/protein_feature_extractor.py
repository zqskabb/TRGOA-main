from torch_geometric.nn import GATConv,global_mean_pool,SAGPooling,global_max_pool, global_add_pool,EdgePooling,GCNConv
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Sequential, ReLU, Sigmoid
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, to_dense_batch
from torch_geometric.nn.inits import uniform, reset
from torch_geometric.nn import (global_add_pool, JumpingKnowledge)
from soft_mask_gnn import WeightConv1, SparseConv, WeightConv2


class_length = {"bp": 19462, "mf": 6239, "cc":2434, "all":2752}

class Protein_feature_extraction(torch.nn.Module):
    def __init__(self, 
                 num_features = 1280,
                 num_classes = 2752,  
                 num_layers = 4,
                 hidden = 256,
                 weight_conv='WeightConv1',
                 multi_channel='True'):
        super(Protein_feature_extraction, self).__init__()
        self.lin0 = Linear(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SparseConv(hidden, hidden))

        self.masks = torch.nn.ModuleList()
        if multi_channel == 'True':
            out_channel = hidden
        else:
            out_channel = 1
        if weight_conv != 'WeightConv2':
            for i in range(num_layers):
                self.masks.append(WeightConv1(hidden, hidden, out_channel))
        else:
            for i in range(num_layers):
                self.masks.append(WeightConv2(Sequential(
                    Linear(hidden * 2, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, out_channel),
                    Sigmoid()
                )))
        self.jump = JumpingKnowledge(mode='cat')

    def reset_parameters(self):
        self.lin0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for mask in self.masks:
            mask.reset_parameters()

    def forward(self, seq, edge_index, esm_token, batch):
        esm_token = self.lin0(esm_token.unsqueeze(0))
        mask_val = None
        xs = []
        for i, conv in enumerate(self.convs):
            mask = self.masks[i]
            mask_val = mask(esm_token, edge_index, mask_val)
            esm_token = F.relu(conv(esm_token, edge_index, mask_val))

            xs += esm_token
        esm_token = self.jump(xs)
        residue_embed, mask = to_dense_batch(esm_token, batch)    
        return residue_embed, mask

    def __repr__(self):
        return self.__class__.__name__

