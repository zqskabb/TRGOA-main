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



class Protein_Label_att(torch.nn.Module):
   
    def __init__(self):
        super(Protein_Label_att, self).__init__()
    def forward(self, protein_embedding, label_embedding, mask):

        mask = mask.unsqueeze(1).expand((protein_embedding.shape[0], label_embedding.shape[1], protein_embedding.shape[1]))
        protein_embedding = protein_embedding.float()
        attention = torch.matmul(label_embedding, protein_embedding.permute(0, 2, 1))
        attention = attention* protein_embedding.size(-1) ** -0.5
        attention = attention.masked_fill_(mask == 0., torch.tensor(-1e9, dtype=torch.float32))
        attention = torch.sigmoid(attention)
        return attention
    
 
