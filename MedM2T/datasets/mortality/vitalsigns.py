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

class VitalsNumDataset(Dataset):
    def __init__(self, vitals, targets, ids):
        if type(vitals) is np.ndarray:
            self.vitals = torch.from_numpy(vitals).type(torch.FloatTensor)
        else:
            self.vitals = torch.FloatTensor(vitals)
        self.targets = torch.LongTensor(targets)
        self.ids = torch.LongTensor(ids)

    def __getitem__(self, index):
        return self.vitals[index], self.targets[index], self.ids[index]

    def __len__(self):
        return self.targets.size(0)

class VitalsCatDataset(Dataset):
    def __init__(self, vitals_values, vitals_sources, targets, ids):
        vitals_vals_tokens = []
        vitals_vals_size = []
        for tw_tokens in vitals_values:
            vitals_vals_tokens.append([torch.tensor(val_tokens, dtype=torch.int64) for val_tokens in tw_tokens])
            vitals_vals_size.append([len(val_tokens) for val_tokens in tw_tokens])

        vitals_srcs_tokens = []
        vitals_srcs_size = []
        for tw_tokens in vitals_sources:
            vitals_srcs_tokens.append([torch.tensor(src_tokens, dtype=torch.int64) for src_tokens in tw_tokens])
            vitals_srcs_size.append([len(src_tokens) for src_tokens in tw_tokens])
        
        self.vitals_vals_tokens = vitals_vals_tokens
        self.vitals_vals_size = vitals_vals_size
        self.vitals_srcs_tokens = vitals_srcs_tokens
        self.vitals_srcs_size = vitals_srcs_size
        self.targets = torch.LongTensor(targets)
        self.ids = torch.LongTensor(ids)

    def __getitem__(self, index):
        return [self.vitals_vals_tokens[index], self.vitals_vals_size[index]], [self.vitals_srcs_tokens[index], self.vitals_srcs_size[index]], self.targets[index], self.ids[index]

    def __len__(self):
        return self.targets.size(0)

class VitalsNumLoader():
    def __init__(self, ids, targets, win_size = None, seg_num = None):
        self.ids = ids
        self.targets = targets
        #read data
        self.fname = "mimic_first_icu_cont_vitals_6mins.pkl"
        with open(data_path+self.fname, 'rb') as f:
            subject_vitals = pickle.load(f)

        self.vitals_shape = subject_vitals.get(list(subject_vitals.keys())[0]).shape
        self.vitals_None = np.zeros(self.vitals_shape)
        self.vitals_items = ['220045', '220179', '220180', '220181', \
                             '223762', '220210', '220277', '223835']

        self.valid_idx = []
        self.vitals = []
        for i, id in enumerate(self.ids):
            if id in subject_vitals:
                self.valid_idx.append(i)
                self.vitals.append(subject_vitals.get(id))
            else:
                self.vitals.append(self.vitals_None)
        self.valid_idx = np.array(self.valid_idx)
        self.valid_ids = self.ids[self.valid_idx]
        self.vitals = np.array(self.vitals)
        if win_size is not None and seg_num is not None:
            self.vitals_segs = self.set_seg_data(win_size, seg_num)
        else:
            self.vitals_segs = None
            
    def __len__(self):
        return len(self.ids)
    
    def get_ids(self, only_valid = False):
        if only_valid:
            return self.valid_ids
        return self.ids

    def get_shift(self, sig_len, win_size, seg_num):
        return (sig_len - win_size)/(seg_num-1)
    
    def split_segs(self, vitals, win_size, shift_size):
        seg_vitals = []
        vitals_len = self.vitals_shape[1]
        for idx in np.arange(0, vitals_len-win_size+1, shift_size):
            idx = int(idx)
            seg_vitals.append(vitals[:, idx:idx+win_size])
        return seg_vitals
    
    def get_orig_data(self, only_valid = False):
        if only_valid:
            valid_vitals = self.vitals[self.valid_idx]
            return valid_vitals
        
        return self.vitals
    
    def set_seg_data(self, win_size, seg_num):
        self.vitals_segs = []
        self.win_size = win_size
        self.seg_num = seg_num
        self.shift_size = self.get_shift(self.vitals_shape[1], win_size, seg_num)
        for vitals in self.vitals:
            self.vitals_segs.append(self.split_segs(vitals, self.win_size, self.shift_size))
        self.vitals_segs = np.array(self.vitals_segs)
        self.targets_segs = np.tile(self.targets, (self.seg_num, 1)).T
        self.ids_segs = np.tile(self.ids, (self.seg_num, 1)).T
        self.idx_segs = np.arange(len(self.ids)*self.seg_num).reshape(len(self.ids),self.seg_num)

    def get_data(self, only_valid = False, flatten = False):
        if only_valid:
            vitals = self.vitals_segs[self.valid_idx]
        else:
            vitals = self.vitals_segs
        if flatten:
            return vitals.reshape(-1, *vitals.shape[2:])
        return vitals
    
    def get_dataset(self, only_valid = False, flatten = False):
        if self.vitals_segs is None:
            return None
        
        vitals = self.get_data(only_valid, flatten)
        
        if flatten is False:
            if only_valid:
                return VitalsNumDataset(vitals, self.targets[self.valid_idx], self.valid_ids)
            return VitalsNumDataset(vitals, self.targets, self.ids)
        else:
            targets = self.targets_segs[self.valid_idx].reshape(-1) if only_valid else self.targets_segs.reshape(-1)
            ids = self.ids_segs[self.valid_idx].reshape(-1) if only_valid else self.ids_segs.reshape(-1)
            return VitalsNumDataset(vitals, targets, ids)
    
    def get_orig_dataset(self, only_valid = False):
        if only_valid:
            return VitalsNumDataset(self.vitals[self.valid_idx], self.targets[self.valid_idx], self.valid_ids)
        return VitalsNumDataset(self.vitals, self.targets, self.ids)
        
    def get_flatten_idx(self, idx):
        return self.idx_segs[idx].reshape(-1)
    

class VitalsCatLoader():
    def __init__(self, ids, targets):
        self.ids = ids
        self.targets = targets

        #Category
        self.fname = "mimic_icu24_vitals_24bins.pkl"
        with open(data_path+self.fname, 'rb') as f:
            subject_vals = pickle.load(f)

        self.win_num = len(subject_vals.get(list(subject_vals.keys())[0]))

        self.cats_dict_fname = "mimic_first_icu_vitals_label_dict.pkl"
        with open(data_path+self.cats_dict_fname, 'rb') as f:
            vals_items_dict = pickle.load(f)

        #get vals max label
        max_label = 0
        for v in vals_items_dict.values():
            max_label = max(max_label, max(v["labels"]))
        self.vals_None_label = max_label+1
        self.vals_None = [[self.vals_None_label] for _ in range(self.win_num)]

        #get sources 
        srcs_items_dict = {k:v for k,v in zip(vals_items_dict.keys(), range(len(vals_items_dict)))}
        self.srcs_None_label = len(srcs_items_dict)
        self.srcs_None = [[self.srcs_None_label] for _ in range(self.win_num)]

        #mapping subject_vals to subject_srcs
        val2src = {}
        for k,v in vals_items_dict.items():
            val2src.update({val_label:srcs_items_dict[k] for val_label in v["labels"]})
        val2src[self.vals_None_label] = self.srcs_None_label

        subject_srcs = {}
        for subject_id, vals_win_tokens in subject_vals.items():
            subject_srcs[subject_id] = [[val2src[val] for val in vals] for vals in vals_win_tokens]

        self.values = []
        self.sources = []
        self.valid_idx = []
        for idx, id in enumerate(self.ids):
            if id in subject_vals:
                self.values.append(subject_vals.get(id))
                self.sources.append(subject_srcs.get(id))
                self.valid_idx.append(idx)
            else:
                self.values.append(self.vals_None)
                self.sources.append(self.srcs_None)
        
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
            return VitalsCatDataset(valid_values, valid_sources, valid_targets, self.valid_ids)
        else:
            return VitalsCatDataset(self.values, self.sources, self.targets, self.ids)
        
def multiscale_vitalsigns(vitals1, vitals2):
    vitals = np.concatenate([vitals1.vitals, vitals2.vitals], axis=-1)
    assert(torch.equal(vitals1.targets, vitals2.targets))
    assert(torch.equal(vitals1.ids, vitals2.ids))
    targets = vitals1.targets
    ids = vitals1.ids
    return VitalsNumDataset(vitals, targets, ids)


class VitalsStaticDataset():
    def __init__(self, vitals, targets, ids):
        self.vitals = torch.from_numpy(vitals).type(torch.FloatTensor)
        self.targets = torch.LongTensor(targets)
        self.ids = torch.LongTensor(ids)

    def __getitem__(self, index):
        return self.vitals[index], self.targets[index], self.ids[index]

    def __len__(self):
        return self.targets.size(0)
    
class VitalsStaticLoader():
    def __init__(self, ids, targets):
        self.ids = ids
        self.targets = targets
        #read data
        self.num_fname = "vitals_num_stats_df.pkl"
        self.cat_fname = "vitals_cat_stats_df.pkl"
        vitals_num_df = pd.read_pickle(data_path+self.num_fname)
        vitals_cat_df = pd.read_pickle(data_path+self.cat_fname)

        self.num_data_cols = vitals_num_df.columns
        self.cat_data_cols = vitals_cat_df.columns

        vitals_df = pd.concat([vitals_num_df, vitals_cat_df], axis=1)
        self.valid_idx = np.where(np.isin(ids, vitals_df.index))[0]
        self.valid_ids = ids[self.valid_idx]
        vitals_df = vitals_df.reindex(ids)
        vitals_df = vitals_df.fillna(0)
        self.data = vitals_df.values

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
            return VitalsStaticDataset(self.data[self.valid_idx], self.targets[self.valid_idx], self.valid_ids)
        return VitalsStaticDataset(self.data, self.targets, self.ids)

class VitalsDataset(Dataset):
    def __init__(self, ids, targets, vitals_num, vitals_cat):
        self.ids = ids
        self.targets = torch.LongTensor(targets)
        self.vitals_num = vitals_num
        self.vitals_cat = vitals_cat

    def __getitem__(self, index):
        _data = [self.vitals_num[index],
                 self.vitals_cat[index],
                 self.targets[index], 
                 self.ids[index]]
        return _data

    def __len__(self):
        return self.targets.size(0)