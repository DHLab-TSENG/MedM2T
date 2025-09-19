from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
from src.config import device

def tokens_batch(tokens_list, accum, onset = True):
    _tokens_list = []
    _onset_list = [0]
    for tokens, tokens_size in tokens_list:
        if accum:
            _tokens_list.append(torch.concat(tokens))
            _onset_list.append(sum(tokens_size)) 
        else:
            _tokens_list.append(tokens)
            _onset_list.append(tokens_size)
    
    _tokens_list = torch.cat(_tokens_list).to(device)
    if onset:
        _onset_list = torch.tensor(_onset_list[:-1]).cumsum(dim=0).to(device)
    else:
        ## get size
        _onset_list = torch.tensor(_onset_list[1:]).to(device)
    return [_tokens_list, _onset_list]


def time_win_tokens_batch(tokens_tw_list, accum, onset = True):
    """
    Concatenate tokens in each time window
    Args:
        tokens_tw_list: list of tuple (tokens_tw, tokens_size)
        accum: bool, whether to accumulate tokens
        onset: bool, whether to return onset
    Returns:
        tokens_list: list of tensor, tokens in each time window
        onset_list: list of tensor, onset in each time window
    """
    tokens_list = None
    onset_list = None

    win_num = 0
    for tokens_tw, tokens_size in tokens_tw_list:
        if tokens_list is None:
            win_num = len(tokens_tw)
            tokens_list = [[] for _ in range(win_num)]
            onset_list = [[0] for _ in range(win_num)]
            
        for i in range(win_num):
            if accum:
                tokens_list[i].append(torch.concat(tokens_tw[:i+1]))
                onset_list[i].append(sum(tokens_size[:i+1])) 
            else:
                tokens_list[i].append(tokens_tw[i])
                onset_list[i].append(tokens_size[i])
    
    for win_i in range(win_num):
        tokens_list[win_i] = torch.cat(tokens_list[win_i]).to(device)
        if onset:
            onset_list[win_i] = torch.tensor(onset_list[win_i][:-1]).cumsum(dim=0).to(device)
        else:
            ## get size
            onset_list[win_i] = torch.tensor(onset_list[win_i][1:]).to(device)

    return [tokens_list, onset_list]

def ecg_bags_batch(bags_list):
    sig_list = []
    feats_list = []
    tokens_list = []
    tokens_onset_list = [0]
    batch_i = []
    win_i = []
    batch_num = len(bags_list)
    for i, (sig, feats, note_tokens, win) in enumerate(bags_list):
        sig_list.extend(sig)
        feats_list.extend(feats)
        tokens_list.extend(note_tokens[0])
        tokens_onset_list.extend(note_tokens[1])
        win_i.extend(win)
        batch_i.extend([i]*len(win))
        
    sig_list = torch.stack(sig_list).to(device)
    feats_list = torch.stack(feats_list).to(device)
    tokens_list = torch.stack(tokens_list).to(device)
    tokens_onset_list = torch.tensor(tokens_onset_list[:-1]).cumsum(dim=0).to(device)
    batch_i = torch.tensor(batch_i).to(device)
    win_i = torch.tensor(win_i).to(device)

    return (sig_list, feats_list, (tokens_list, tokens_onset_list)), batch_num, batch_i, win_i

def basic_collate_fn(data_list):
    return torch.stack(data_list).to(device)

class CustomCollateFn:
    def __init__(self, modalities_num, tokens_idx = [], accum = True, onset = True ,
                multi_idx = [], merge_idx_list = [], tokens_tw = True):
        """
        Args:
            modalities_num: int, number of modalities
            tokens_idx: list of int, index of time window tokens
            multi_idx: list of int, index of multi input
            merge_idx_list: list of list of int, merge modalities
            accum: bool, whether to accumulate tokens for time window tokens
            onset: bool, whether to return onset for time window tokens
        Returns:
            collate_fn: function, collate function
            """
        self.modalities_num = modalities_num
        self.tokens_idx = tokens_idx
        self.tokens_tw = tokens_tw
        self.multi_idx = multi_idx
        self.merge_idx_list = merge_idx_list
        self.accum = accum
        self.onset = onset

    def __call__(self, batch):
        label_list = []
        index_list = []
        modalities_list = [ [] for _ in range(self.modalities_num)]
        
        #data: (*modalities, label, index)
        for data in batch:
            for i, modality in enumerate(data[:-2]):
                modalities_list[i].append(modality)
            label_list.append(data[-2])
            index_list.append(data[-1])

        label_list = torch.tensor(label_list, dtype=torch.int64).to(device)
        for i in range(self.modalities_num):
            if i in self.tokens_idx:
                if self.tokens_tw:
                    modalities_list[i] = time_win_tokens_batch(modalities_list[i], self.accum, self.onset)
                else:
                    modalities_list[i] = tokens_batch(modalities_list[i], self.accum, self.onset)
            elif i in self.multi_idx:
                multi_input_list = []
                num_input = len(modalities_list[i][0])
                for in_i in range(num_input):
                    input_list = []
                    for data in modalities_list[i]:
                        input_list.append(data[in_i])
                    multi_input_list.append(torch.stack(input_list).to(device))
                modalities_list[i] = multi_input_list
            else:
                modalities_list[i] = torch.stack(modalities_list[i]).to(device)

        if len(self.merge_idx_list):
            rm_idx_list = []
            for merge_idx in self.merge_idx_list:
                _data = [modalities_list[i] for i in merge_idx]
                modalities_list[merge_idx[0]] = _data
                rm_idx_list.extend(merge_idx[1:])

            for i in sorted(rm_idx_list, reverse=True):
                del modalities_list[i]
        
        return (*modalities_list, label_list, index_list)
    
class CreateCustomDataset(Dataset):
    def __init__(self, modalities_num, collate_fn_params, classfication = True):
        self.modalities_num = modalities_num
        self.collate_fn_params = collate_fn_params
        self.classfication = classfication
    
    def collate_fn(self, param, modality):
        _collate_fn = globals()[param["name"]]
        if "param" in param:
            return _collate_fn(modality, **param["param"])
        else:
            return _collate_fn(modality)

    def __call__(self, batch):
        target_list = []
        index_list = []
        modalities_list = [ [] for _ in range(self.modalities_num)]
        
        #data: (*modalities, label, index)
        for data in batch:
            for i, modality in enumerate(data[:-2]):
                modalities_list[i].append(modality)
            target_list.append(data[-2])
            index_list.append(data[-1])

        if self.classfication:
            target_list = torch.tensor(target_list, dtype=torch.int64).to(device)
        else:
            target_list = torch.tensor(target_list, dtype=torch.float).to(device)
        
        for i, param in enumerate(self.collate_fn_params):
            if type(param) == list:
                _modalities_list = []
                for j, _param in enumerate(param):
                    _modality = [data[j] for data in modalities_list[i]]
                    _modalities_list.append(self.collate_fn(_param, _modality))
                modalities_list[i] = _modalities_list
            else:
                modalities_list[i] = self.collate_fn(param, modalities_list[i])

        return (*modalities_list, target_list, index_list)