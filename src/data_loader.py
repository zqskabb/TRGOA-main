import torch
import numpy as np

import numpy as np
import os
from torch_geometric.data import Data, Dataset, DataLoader
import random
# ["ac", "seq", "edge_index", "esm_toke", "esm_representation", "label"]

# esm_rep, seq, contact, pssm, seq_embed = x(esm_toke), seq(seq), edge_index(edge_index), pssm(无), seq_embed(ESM_respresentation) 

class contact_data(Data):
    def __cat_dim__(self, key, item, store):
        if key in ['esm_representation', 'label', 'chain_id']:
            return None
        else:
            return super().__cat_dim__(key, item)

class Protein_ours_data(Dataset):
    def __init__(self, root, chain_list, transform = None, pre_transform = None):
        super(Protein_ours_data, self).__init__(root, transform, pre_transform)
        self.chain_list = open(chain_list).readlines()
        self.chain_ids = [chain.strip() for chain in self.chain_list]
        print(len(self.chain_ids))
    @property
    def raw_file_names(self):
        return self.chain_ids
    
    def len(self):
        return len(self.chain_ids)

    def get(self,idx):
        data = torch.load(self.root + '/' + self.chain_ids[idx] + '.pt')
        data = contact_data(esm_token = data['esm_token'], seq = data['seq'], edge_index = data['edge_index'], esm_representation = data['esm_representation'], label = data['label'], chain_id = self.chain_ids[idx])
        return data


class Protein_CAFA3_data(Dataset):
    def __init__(self, root, chain_list, transform = None, pre_transform = None):
        super(Protein_CAFA3_data, self).__init__(root, transform, pre_transform)
        self.chain_list = open(chain_list).readlines()
        self.chain_ids = [chain.strip() for chain in self.chain_list]
    @property
    def raw_file_names(self):
        return self.chain_ids
    
    def len(self):
        return len(self.chain_ids)

    def get(self,idx):
        data = torch.load(self.root + '/' + self.chain_ids[idx] + '.pt')
        if type(data['edge_index'])==torch.Tensor:
            data = contact_data(esm_token = data['esm_token'], seq = data['seq'], edge_index = data['edge_index'], esm_representation = data['esm_representation'], label = data['label'], chain_id = self.chain_ids[idx])
        else:
            data = contact_data(esm_token = data['esm_token'], seq = data['seq'], edge_index = random.choice(data['edge_index']), esm_representation = data['esm_representation'], label = data['label'], chain_id = self.chain_ids[idx])
        #print(data['esm_token'].shape, data['seq'].shape, len(data['edge_index']),random.choice(data['edge_index']).shape, data['esm_representation'].shape, data['label'].shape, self.chain_ids[idx])
        return data


class Protein_CAFA3(Dataset):
    def __init__(self, root, chain_list, transform = None, pre_transform = None):
        super(Protein_CAFA3, self).__init__(root, transform, pre_transform)
        self.chain_ids = chain_list
    @property
    def raw_file_names(self):
        return self.chain_ids
    
    def len(self):
        return len(self.chain_ids)

    def get(self,idx):
        data = torch.load(self.root + '/' + self.chain_ids[idx] + '.pt')
        if type(data['edge_index'])==torch.Tensor:
            data = contact_data(esm_token = data['esm_token'], seq = data['seq'], edge_index = data['edge_index'], esm_representation = data['esm_representation'], label = data['label'], chain_id = self.chain_ids[idx])
        else:
            data = contact_data(esm_token = data['esm_token'], seq = data['seq'], edge_index = random.choice(data['edge_index']), esm_representation = data['esm_representation'], label = data['label'], chain_id = self.chain_ids[idx])
        #print(data['esm_token'].shape, data['seq'].shape, len(data['edge_index']),random.choice(data['edge_index']).shape, data['esm_representation'].shape, data['label'].shape, self.chain_ids[idx])
        return data

class Protein_CAFA3_pickle(Dataset):
    def __init__(self, root_file, chain_list, transform = None, pre_transform = None):
        super(Protein_CAFA3_pickle, self).__init__(root_file, transform, pre_transform)
        self.chain_ids = chain_list
        self.root_file = root_file
    @property
    def raw_file_names(self):
        return self.chain_ids
    
    def len(self):
        return len(self.chain_ids)

    def get(self,idx):
        data = self.root_file[int(self.chain_ids[idx])-1]
        if type(data['edge_index'])==torch.Tensor:
            data = contact_data(esm_token = data['esm_token'], seq = data['seq'], edge_index = data['edge_index'], esm_representation = data['esm_representation'], label = data['label'], chain_id = self.chain_ids[idx])
        else:
            data = contact_data(esm_token = data['esm_token'], seq = data['seq'], edge_index = random.choice(data['edge_index']), esm_representation = data['esm_representation'], label = data['label'], chain_id = self.chain_ids[idx])
        #print(data['esm_token'].shape, data['seq'].shape, len(data['edge_index']),random.choice(data['edge_index']).shape, data['esm_representation'].shape, data['label'].shape, self.chain_ids[idx])
        return data






class contact_data_1(Data):
    def __cat_dim__(self, key, item, store):
        if key in ['seq_embed', 'label', 'chain_id']:
            return None
        else:
            return super().__cat_dim__(key, item)


class Protein_Gnn_data(Dataset):
    def __init__(self, root, chain_list, transform = None, pre_transform = None):
        super(Protein_Gnn_data, self).__init__(root, transform, pre_transform)
        self.chain_list = open(chain_list).readlines()
        self.chain_ids = [chain.strip().upper() for chain in self.chain_list]#[:11]
    @property
    def raw_file_names(self):
        return self.chain_ids
    
    def len(self):
        return len(self.chain_ids)

    def get(self,idx):
        files_name = os.listdir(self.root)
        file_dict = {name.split(".")[0].upper()+'.pt': name for name in files_name}

        data = torch.load(self.root + '/' + file_dict[self.chain_ids[idx]+ '.pt'])
        data = contact_data_1(x = data['x'], pssm = data['pssm'].T, seq = data['seq'].T, edge_index = data['edge_index'], seq_embed = data['seq_embed'], label = data['label'], chain_id = self.chain_ids[idx])
        return data
    
def index_select(train_chain, test_chain, train_num, valid_num):
    train_chain = open(train_chain).readlines()
    chain_ids = [chain.strip().upper() for chain in train_chain]
    print(len(chain_ids))

    valid_index = random.sample(list(range(train_num)),valid_num)
    train_index = list(set(list(range(train_num)))-set(valid_index))

    X_train = [chain_ids[i] for i in train_index]
    X_valid = [chain_ids[i] for i in valid_index]

    test_chain = open(test_chain).readlines()
    X_test = [chain.strip().upper() for chain in test_chain]
    return X_train, X_valid, X_test