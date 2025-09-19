from torch.utils.data import Dataset
import torch
class MultiModalDataset(Dataset):
    def __init__(self, ids, targets, static, labs, ecg):
        self.ids = ids
        self.targets = torch.LongTensor(targets)
        self.static = static
        self.labs = labs
        self.ecg = ecg

    def __getitem__(self, index):
        _data = [self.static[index], 
                 self.labs[index], 
                 self.ecg[index], 
                 self.targets[index], 
                 self.ids[index]]
        return _data

    def __len__(self):
        return self.targets.size(0)