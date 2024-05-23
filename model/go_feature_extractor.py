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


class Label_feature_extraction(torch.nn.Module):  

    def __init__(self, 
                 ont= "all",
                 hidden_channels = [512, 1024]):
        super(Label_feature_extraction, self).__init__()
        self.ont = ont
        self.embed_channels = class_length[self.ont]
        self.hidden_channels = hidden_channels

        self.gc = nn.Sequential(
            nn.Conv1d(self.embed_channels, self.hidden_channels[0], kernel_size=1),
            nn.ReLU(inplace=True)
        )

    
        self.gat = GATConv(self.hidden_channels[0], self.hidden_channels[1])
    def forward(self, label_relationship, protein_embedding): 
        label_embedding = torch.eye(self.embed_channels, device = label_relationship.device, dtype=torch.float32)
        out = self.gc(label_embedding)
        out = self.gat(out.T, label_relationship)
        out = out.expand(protein_embedding.shape[0], out.shape[0], out.shape[1])
        return out

