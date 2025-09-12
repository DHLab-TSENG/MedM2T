from .config import data_path
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset
import torch

class ECGSigDataset(Dataset):
    def __init__(self, sig, targets, ids):
        self.sig = torch.from_numpy(sig).type(torch.FloatTensor)
        self.targets = torch.LongTensor(targets)
        self.ids = ids
    
    def __getitem__(self, index):
        return self.sig[index], self.targets[index], self.ids[index]
    
    def __len__(self):
        return self.targets.size(0)
    
class ECGFeatsDataset(Dataset):
    def __init__(self, feats, targets, ids):
        self.feats = torch.from_numpy(feats).type(torch.FloatTensor)
        self.targets = torch.LongTensor(targets)
        self.ids = ids
    
    def __getitem__(self, index):
        return self.feats[index], self.targets[index], self.ids[index]
    
    def __len__(self):
        return self.targets.size(0)
    
class ECGNoteDataset(Dataset):
    def __init__(self, note_tokens, targets, ids):
        tokens = []
        tokens_size = []
        for token in note_tokens:
            tokens.append(torch.tensor(token))
            tokens_size.append(len(token))
        self.note_tokens = tokens
        self.note_tokens_size = torch.tensor(tokens_size)
        self.targets = torch.LongTensor(targets)
        self.ids = ids
    
    def __getitem__(self, index):
        return (self.note_tokens[index], self.note_tokens_size[index]), self.targets[index], self.ids[index]
    
    def __len__(self):
        return self.targets.size(0)
    
class ECGFusionDataset(Dataset):
    def __init__(self, sig, feats, note_tokens, targets, ids):
        self.ecg_sig = torch.from_numpy(sig).type(torch.FloatTensor)
        self.ecg_feats = torch.from_numpy(feats).type(torch.FloatTensor)
        tokens = []
        tokens_size = []
        for token in note_tokens:
            tokens.append(torch.tensor(token))
            tokens_size.append(len(token))
        self.note_tokens = tokens
        self.note_tokens_size = torch.tensor(tokens_size)
        self.targets = torch.LongTensor(targets)
        self.ids = ids

    def __getitem__(self, index):
        return self.ecg_sig[index], self.ecg_feats[index], (self.note_tokens[index], self.note_tokens_size[index]), self.targets[index], self.ids[index]

    def __len__(self):
        return self.targets.size(0)
    
class ECGDataset(Dataset):
    def __init__(self, ecg_bags, targets, ids):
        self.ecg_bags = ecg_bags
        self.targets = torch.LongTensor(targets)
        self.ids = ids

    def __getitem__(self, index):
        return self.ecg_bags[index], self.targets[index], self.ids[index]

    def __len__(self):
        return self.targets.size(0)
    
class ECGMicroEmbsDataset(Dataset):
    def __init__(self, embs, targets, ids):
        self.embs = torch.from_numpy(embs).type(torch.FloatTensor)
        self.targets = torch.LongTensor(targets)
        self.ids = ids

    def __getitem__(self, index):
        return self.embs[index], self.targets[index], self.ids[index]

    def __len__(self):
        return self.targets.size(0)

class ECGLoader():
    def __init__(self, ids, targets, tw_pth_list = [20, 40, 60, 80]):
        self.ids = ids
        self.targets = targets
        id2target = dict(zip(ids, targets))

        #read subject dict
        #key: subject_id, value: {"ecg_id":[ecg_id], "time_diff(hours)":[time_diff]}
        self.sub_dict_fname = "24hr_icu_ecg_dict.pkl"
        with open(data_path+self.sub_dict_fname, 'rb') as f:
            self.subject_dict = pickle.load(f)

        self.valid_idx = np.where(np.isin(self.ids, list(self.subject_dict.keys())))[0]
        self.valid_ids = self.ids[self.valid_idx]
        
        #ecg2sub: ecg_id to subject_id and target
        self.ecg2sub = {}
        ecg_time_diff = []
        for id, ecgs in self.subject_dict.items():
            for ecg_id in ecgs["ecg_id"]:
                self.ecg2sub[ecg_id] = {"subject_id": id, "target": id2target[id]}
            self.subject_dict[id]["target"] = id2target[id]
            ecg_time_diff.extend(ecgs["time_diff(hours)"])
        self.ecg_time_diff = np.array(ecg_time_diff)
        self.set_time_bins(tw_pth_list)

        #read ecg ids
        self.ecg_ids = np.load(data_path+"24hr_icu_ecg_id.npy")
        self.ecg_targets = np.array([self.ecg2sub[ecg_id]["target"] for ecg_id in self.ecg_ids])
        self.ecg_ids2idx = dict(zip(self.ecg_ids, range(len(self.ecg_ids)))) 

        #read signal, order by ecg_id
        self.sig_fname = "24hr_icu_ecg.npy"
        ecg_signal = np.load(data_path+self.sig_fname)
        self.ecg_signal = ecg_signal[:,:,300:940]
        self.sigs_shape = self.ecg_signal[0].shape
        self.ecg_signal_None = np.zeros(self.sigs_shape)
        
        #read statement, transform to tokens
        self.statement_fname = "24hr_icu_ecg_statement_df.pkl"
        ecg_statement_df = pd.read_pickle(data_path+self.statement_fname)
        self.ecg_statement_cols = ecg_statement_df.columns[1:]
        ecg_statement_df = ecg_statement_df.loc[self.ecg_ids][self.ecg_statement_cols]
        self.ecg_statement_None_label = len(self.ecg_statement_cols)
        self.ecg_statement_tokens_None = [self.ecg_statement_None_label]
        ecg_statement_df.columns = range(self.ecg_statement_None_label)
        ecg_statement_tokens = ecg_statement_df.apply(lambda x: x[x == 1].index.tolist(), axis=1)
        ecg_statement_tokens = ecg_statement_tokens.apply(lambda x: x if len(x) > 0 else self.ecg_statement_tokens_None)
        self.ecg_statement_tokens = ecg_statement_tokens.values
        
        #read features
        ecg_reports = pd.read_csv(data_path+"machine_measurements.csv", low_memory = False)
        ecg_reports.ecg_time = pd.to_datetime(ecg_reports.ecg_time)
        ecg_reports.index = ecg_reports.study_id
        self.ecg_feats_cols = ['rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis']
        ecg_feats = ecg_reports.loc[self.ecg_ids][self.ecg_feats_cols]
        ecg_feats = ecg_feats.fillna(0)
        self.ecg_feats = ecg_feats.values
        self.ecg_feats_None = np.zeros(len(self.ecg_feats_cols))

        self.data = None

    def __len__(self, only_valid = False):
        if only_valid:
            return len(self.valid_ids)
        return len(self.ids)
    
    def get_ids(self, only_valid = False):
        if only_valid:
            return self.valid_ids
        return self.ids
    
    def get_ecg_ids(self, return_target = False):
        if return_target:
            return self.ecg_ids, self.ecg_targets
        return self.ecg_ids
    
    def set_time_bins(self, tw_pth_list = [20, 40, 60, 80]):
        self.tw_pth_list = tw_pth_list
        ecg_time_bins = np.percentile(self.ecg_time_diff, tw_pth_list)
        ecg_time_bins = np.unique(ecg_time_bins)
        self.ecg_time_bins = np.concatenate([[-np.inf],ecg_time_bins,[np.inf]])
        self.ecg_win_i = np.arange(len(self.ecg_time_bins)-1)
    
    def get_ecg_data(self, data_type, last=False, only_valid=False):
        if data_type == "sig":
            data =  self.ecg_signal
            data_None = self.ecg_signal_None
        elif data_type == "feats":
            data = self.ecg_feats
            data_None = self.ecg_feats_None
        elif data_type == "tokens":
            data = self.ecg_statement_tokens
            data_None = self.ecg_statement_tokens_None
        
        if last:
            _data = []
            ids = self.valid_ids if only_valid else self.ids
            for id in ids:
                if id in self.subject_dict:
                    last_ecg = self.subject_dict[id]["ecg_id"][-1]
                    idx = self.ecg_ids2idx[last_ecg]
                    _data.append(data[idx])
                else:
                    _data.append(data_None)
            return _data
        return data
    
    def get_bag_data(self, subject_id):
        if subject_id in self.subject_dict:
            ecg_ids = self.subject_dict[subject_id]["ecg_id"]
            ecg_wins = list(pd.cut(self.subject_dict[subject_id]["time_diff(hours)"], 
                            bins = self.ecg_time_bins, labels = self.ecg_win_i))
            ecg_idxs = [self.ecg_ids2idx[ecg_id] for ecg_id in ecg_ids]
            sig = torch.from_numpy(self.ecg_signal[ecg_idxs]).type(torch.FloatTensor)
            feats = torch.from_numpy(self.ecg_feats[ecg_idxs]).type(torch.FloatTensor)
            tokens = []
            tokens_size = []
            for token in self.ecg_statement_tokens[ecg_idxs]:
                tokens.extend(token)
                tokens_size.append(len(token))
            tokens = torch.tensor(tokens)
            tokens_size = torch.tensor(tokens_size)
        else:
            sig = torch.from_numpy(self.ecg_signal_None).type(torch.FloatTensor).unsqueeze(0)
            feats = torch.from_numpy(self.ecg_feats_None).type(torch.FloatTensor).unsqueeze(0)
            tokens = torch.tensor(self.ecg_statement_tokens_None)
            tokens_size = torch.tensor([1])
            ecg_wins = []

        return sig, feats, (tokens, tokens_size), ecg_wins

    def get_data(self, only_valid = False):
        if self.data is None:
            data_list = []
            for id in self.ids:
                sig, feats, tokens, ecg_wins = self.get_bag_data(id)
                data_list.append((sig, feats, tokens, ecg_wins))
            self.data = data_list
        if only_valid:
            return [self.data[i] for i in self.valid_idx]
        return self.data

    def get_ecg_dataset(self, type):
        if type == "sig":
            return ECGSigDataset(self.ecg_signal, self.ecg_targets, self.ecg_ids)
        elif type == "feats":
            return ECGFeatsDataset(self.ecg_feats, self.ecg_targets, self.ecg_ids)
        elif type == "tokens":
            return ECGNoteDataset(self.ecg_statement_tokens, self.ecg_targets, self.ecg_ids)
        elif type == "fusion":
            return ECGFusionDataset(self.ecg_signal, self.ecg_feats, self.ecg_statement_tokens, self.ecg_targets, self.ecg_ids)
        
    def get_subi2ecgi(self, sub_idx_list, only_valid = True):
        ids = self.valid_ids if only_valid else self.ids
        ecg_idx_list = []
        for sub_idx in sub_idx_list:
            id = ids[sub_idx]
            ecg_idx_list.extend([self.ecg_ids2idx[ecg_id] for ecg_id in self.subject_dict[id]["ecg_id"]])
        return ecg_idx_list
    
    def get_dataset(self, only_valid = False):
        data = self.get_data(only_valid)
        if only_valid:
            return ECGDataset(data, self.targets[self.valid_idx], self.valid_ids)
        else:
            return ECGDataset(data, self.targets, self.ids)