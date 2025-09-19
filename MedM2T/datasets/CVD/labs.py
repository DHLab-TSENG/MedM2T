from .config import data_path
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset
import torch

def get_max_token(wins_values):
    max_token = 0
    for wins in wins_values:
        tokens = []
        for w in wins:
            tokens.extend(w)
        max_token = max(max_token, max(tokens))
    return max_token


class LabsDataset(Dataset):
    def __init__(self, labs_values, labs_sources, targets, ids):
        labs_vals_tokens = []
        labs_vals_size = []
        for tw_tokens in labs_values:
            labs_vals_tokens.append([torch.tensor(val_tokens, dtype=torch.int64) for val_tokens in tw_tokens])
            labs_vals_size.append([len(val_tokens) for val_tokens in tw_tokens])

        labs_srcs_tokens = []
        labs_srcs_size = []
        for tw_tokens in labs_sources:
            labs_srcs_tokens.append([torch.tensor(src_tokens, dtype=torch.int64) for src_tokens in tw_tokens])
            labs_srcs_size.append([len(src_tokens) for src_tokens in tw_tokens])
        
        self.labs_vals_tokens = labs_vals_tokens
        self.labs_vals_size = labs_vals_size
        self.labs_srcs_tokens = labs_srcs_tokens
        self.labs_srcs_size = labs_srcs_size
        self.targets = torch.LongTensor(targets)
        self.ids = torch.LongTensor(ids)

    def __getitem__(self, index):
        return [self.labs_vals_tokens[index], self.labs_vals_size[index]], [self.labs_srcs_tokens[index], self.labs_srcs_size[index]], self.targets[index], self.ids[index]

    def __len__(self):
        return self.targets.size(0)

class LabsLoader():
    def __init__(self, ids, targets):
        self.ids = ids
        self.targets = targets
        #read data
        self.values_fname = "labs_value_t5_10_20_40_80_v10bins.pkl"
        self.sources_fname = "labs_source_t5_10_20_40_80_v10bins.pkl"
        with open(data_path+self.values_fname, 'rb') as f:
            values = pickle.load(f)

        with open(data_path+self.sources_fname, 'rb') as f:
            sources = pickle.load(f)
        
        self.win_num = len(values.get(list(values.keys())[0]))

        self.values_None_label = get_max_token(values.values())
        self.sources_None_label = get_max_token(sources.values())

        values_None = [[self.values_None_label] for _ in range(self.win_num)]
        sources_None = [[self.sources_None_label] for _ in range(self.win_num)]

        values_ids = values.keys()
        self.values = []
        self.sources = []
        self.valid_idx = []
        for idx, id in enumerate(self.ids):
            if id in values_ids:
                self.values.append(values.get(id))
                self.sources.append(sources.get(id))
                self.valid_idx.append(idx)
            else:
                self.values.append(values_None)
                self.sources.append(sources_None)
        
        self.valid_idx = np.array(self.valid_idx)
        self.valid_ids = self.ids[self.valid_idx]

    def __len__(self):
        return len(self.ids)
    
    def get_ids(self, only_valid = False):
        if only_valid:
            return self.valid_ids
        return self.ids
    
    def get_data(self, only_valid = False):
        if only_valid:
            valid_values = [self.values[i] for i in self.valid_idx]
            valid_sources = [self.sources[i] for i in self.valid_idx]
            return valid_values, valid_sources
        
        return self.values, self.sources
    
    def get_dataset(self, only_valid = False):
        if only_valid:
            valid_values = [self.values[i] for i in self.valid_idx]
            valid_sources = [self.sources[i] for i in self.valid_idx]
            valid_targets = [self.targets[i] for i in self.valid_idx]
            return LabsDataset(valid_values, valid_sources, valid_targets, self.valid_ids)
        else:
            return LabsDataset(self.values, self.sources, self.targets, self.ids)

class LabsTimeSeriesDataset(Dataset):
    def __init__(self, labs_timeseries, targets, ids):
        self.labs_timeseries = torch.FloatTensor(labs_timeseries)
        self.targets = torch.LongTensor(targets)
        self.ids = torch.LongTensor(ids)

    def __getitem__(self, index):
        return self.labs_timeseries[index], self.targets[index], self.ids[index]

    def __len__(self):
        return self.targets.size(0)

class LabsTimeSeriesLoader():
    def __init__(self, ids, targets):
        self.ids = ids
        self.targets = targets
        #read data
        self.fname = "labs_data_dict_5_10_20_40_80_forward.pkl"
        with open(data_path+self.fname, 'rb') as f:
            labs_dict = pickle.load(f)

        data_shape = labs_dict.get(list(labs_dict.keys())[0]).shape
        self.items = ['eGFR', 'troponin_T', 'creatinine_kinase', 'creatine_kinase_MB', 'serum_creatinine', 'cholesterol_HDL', 'cholesterol_LDL', 'cholesterol_total']
        self.win_num = data_shape[0]

        labs_None = torch.zeros(data_shape)
        self.valid_idx = []
        labs_list = []
        for i, id in enumerate(ids):
            if id in labs_dict:
                _labs = labs_dict.get(id)
                _labs = torch.from_numpy(_labs).type(torch.FloatTensor)
                self.valid_idx.append(i)
            else:
                _labs = labs_None
            labs_list.append(_labs)
        
        self.data = torch.stack(labs_list)
        self.win_num = data_shape[0]
        
        self.valid_idx = np.array(self.valid_idx)
        self.valid_ids = self.ids[self.valid_idx]

    def __len__(self):
        return len(self.ids)

    def get_ids(self, only_valid = False):
        if only_valid:
            return self.valid_ids
        return self.ids
    
    def get_data(self, only_valid = False):
        if only_valid:
            return self.data[self.valid_idx]
        return self.data
    
    def get_dataset(self, only_valid = False):
        if only_valid:
            return LabsTimeSeriesDataset(self.data[self.valid_idx], self.targets[self.valid_idx], self.valid_ids)
        return LabsTimeSeriesDataset(self.data, self.targets, self.ids)
    

class LabsStaticDataset(Dataset):
    def __init__(self, labs, targets, ids):
        self.labs = torch.from_numpy(labs).type(torch.FloatTensor)
        self.targets = torch.LongTensor(targets)
        self.ids = torch.LongTensor(ids)

    def __getitem__(self, index):
        return self.labs[index], self.targets[index], self.ids[index]

    def __len__(self):
        return self.targets.size(0)

class LabsStaticLoader():
    def __init__(self, ids, targets):
        self.ids = ids
        self.targets = targets
        #read data
        self.fname = "labs_data_stats_df.pkl"
        with open(data_path+self.fname, 'rb') as f:
            labs_df = pickle.load(f)
        
        self.items = ['eGFR', 'troponin_T', 'creatinine_kinase', 'creatine_kinase_MB', 'serum_creatinine', 'cholesterol_HDL', 'cholesterol_LDL', 'cholesterol_total']
        self.feats_cols = labs_df.columns
        self.feats_cols_closest = [item+"_"+col for item in self.items for col in ["closest", "days"]]
        self.valid_idx = np.where(np.isin(ids, labs_df.index))[0]
        self.valid_ids = ids[self.valid_idx]
        labs_df = labs_df.reindex(ids)
        labs_df = labs_df.fillna(0)
        self.data = labs_df.values
        self.data_closest = labs_df[self.feats_cols_closest].values

    def __len__(self):
        return len(self.ids)
        
    def get_ids(self, only_valid = False):
        if only_valid:
            return self.valid_ids
        return self.ids
    
    def get_data(self, only_valid = False, closest = False):
        data = self.data_closest if closest else self.data
        if only_valid:
            return data[self.valid_idx]
        return data
    
    def get_dataset(self, only_valid = False, closest = False):
        data = self.data_closest if closest else self.data
        if only_valid:
            return LabsStaticDataset(data[self.valid_idx], self.targets[self.valid_idx], self.valid_ids)
        return LabsStaticDataset(data, self.targets, self.ids)