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
        self.targets = torch.FloatTensor(targets)
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
        self.values_fname = "mimic_icu24_labs_10bins_20_40_60_80.pkl"
        self.sources_fname = "mimic_icu24_labs_sources_10bins_20_40_60_80.pkl"
        with open(data_path+self.values_fname, 'rb') as f:
            values = pickle.load(f)

        with open(data_path+self.sources_fname, 'rb') as f:
            sources = pickle.load(f)
        
        self.win_num = len(values.get(list(values.keys())[0]))

        self.values_None_label = int(get_max_token(values.values()))
        self.sources_None_label = int(get_max_token(sources.values()))

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
        self.targets = torch.FloatTensor(targets)
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
        self.num_fname = "labs_num_data_dict_10wins_forward.pkl"
        self.cat_fname = "labs_cat_data_dict_10wins_forward.pkl"
        with open(data_path+self.num_fname, 'rb') as f:
            labs_num_dict = pickle.load(f)

        with open(data_path+self.cat_fname, 'rb') as f:
            labs_cat_dict = pickle.load(f)

        num_data_shape = labs_num_dict.get(list(labs_num_dict.keys())[0]).shape
        cat_data_shape = labs_cat_dict.get(list(labs_cat_dict.keys())[0]).shape
        self.num_data_cols = labs_num_dict.get(list(labs_num_dict.keys())[0]).columns
        self.cat_data_cols = labs_cat_dict.get(list(labs_cat_dict.keys())[0]).columns
        self.win_num = num_data_shape[0]

        num_None = torch.zeros(num_data_shape)
        cat_None = torch.zeros(cat_data_shape)
        
        self.valid_idx = []
        labs_list = []
        labs_final_list = []
        for i, id in enumerate(ids):
            if id in labs_num_dict or id in labs_cat_dict:
                if id in labs_num_dict:
                    _labs_num = labs_num_dict.get(id).values
                    _labs_num = torch.from_numpy(_labs_num).type(torch.FloatTensor)
                else:
                    _labs_num = num_None
                
                if id in labs_cat_dict:
                    _labs_cat = labs_cat_dict.get(id).values
                    _labs_cat = torch.from_numpy(_labs_cat).type(torch.LongTensor)
                else:
                    _labs_cat = cat_None
                
                _labs = torch.cat((_labs_num, _labs_cat), dim=1)
                self.valid_idx.append(i)
            else:
                _labs = torch.cat((num_None, cat_None), dim=1)
            labs_list.append(_labs)
            labs_final_list.append(_labs[-1])
        
        self.data = torch.stack(labs_list)
        self.data_final = torch.stack(labs_final_list)
        
        self.valid_idx = np.array(self.valid_idx)
        self.valid_ids = self.ids[self.valid_idx]

    def __len__(self):
        return len(self.ids)

    def get_ids(self, only_valid = False):
        if only_valid:
            return self.valid_ids
        return self.ids
    
    def get_data(self, only_valid = False, only_final = False):
        data = self.data_final if only_final else self.data
        if only_valid:
            return data[self.valid_idx]
        return data
    
    def get_dataset(self, only_valid = False, only_final = False):
        if only_final:
            if only_valid:
                return LabsTimeSeriesDataset(self.data_final[self.valid_idx], self.targets[self.valid_idx], self.valid_ids)
            return LabsTimeSeriesDataset(self.data_final, self.targets, self.ids)
        else:
            if only_valid:
                return LabsTimeSeriesDataset(self.data[self.valid_idx], self.targets[self.valid_idx], self.valid_ids)
            return LabsTimeSeriesDataset(self.data, self.targets, self.ids)

class LabsStaticDataset(Dataset):
    def __init__(self, labs, targets, ids):
        self.labs = torch.from_numpy(labs).type(torch.FloatTensor)
        self.targets = torch.FloatTensor(targets)
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
        self.num_fname = "labs_num_stats_df.pkl"
        self.cat_fname = "labs_cat_stats_df.pkl"
        labs_num_df = pd.read_pickle(data_path+self.num_fname)
        labs_cat_df = pd.read_pickle(data_path+self.cat_fname)

        self.num_data_cols = labs_num_df.columns
        self.cat_data_cols = labs_cat_df.columns

        labs_df = pd.concat([labs_num_df, labs_cat_df], axis=1)
        self.valid_idx = np.where(np.isin(ids, labs_df.index))[0]
        self.valid_ids = ids[self.valid_idx]
        labs_df = labs_df.reindex(ids)
        labs_df = labs_df.fillna(0)
        self.data = labs_df.values

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
            return LabsStaticDataset(self.data[self.valid_idx], self.targets[self.valid_idx], self.valid_ids)
        return LabsStaticDataset(self.data, self.targets, self.ids)