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
    
 
class TotalModel(nn.Module):
  
    def __init__(self, Protein_feature_extraction, Label_feature_extraction, Protein_Label_att, GnnPF_Model):
        super(TotalModel, self).__init__()
        self.Protein_feature_extraction = Protein_feature_extraction
        self.Label_feature_extraction = Label_feature_extraction
        self.Protein_Label_att = Protein_Label_att
        self.GnnPF_Model = GnnPF_Model

    def forward(self, seq, edge_index, esm_token, esm_representation, batch, label_relationship):
        protein_embedding, mask = self.Protein_feature_extraction(seq, edge_index, esm_token, batch)
        label_embedding = self.Label_feature_extraction(label_relationship, protein_embedding) 
        att = self.Protein_Label_att(protein_embedding, label_embedding, mask)
        out = self.GnnPF_Model(protein_embedding, label_embedding, att, esm_representation)
        return out

    


