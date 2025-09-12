from torch.utils.data import Dataset
import torch
class MultiModalDataset(Dataset):
    def __init__(self, ids, targets, static, labs, vitals_num, vitals_cat, ecg):
        self.ids = ids
        self.targets = torch.LongTensor(targets)
        self.static = static
        self.labs = labs
        self.vitals_num = vitals_num
        self.vitals_cat = vitals_cat
        self.ecg = ecg

    def __getitem__(self, index):
        _data = [self.static[index], 
                 self.labs[index], 
                 self.vitals_num[index],
                 self.vitals_cat[index],
                 self.ecg[index], 
                 self.targets[index], 
                 self.ids[index]]
        return _data

    def __len__(self):
        return self.targets.size(0)