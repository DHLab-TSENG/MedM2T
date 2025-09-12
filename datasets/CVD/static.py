from .config import data_path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class StaticDataset(Dataset):
    def __init__(self, static, targets, ids):
        self.static = torch.from_numpy(static).type(torch.FloatTensor)
        self.targets = torch.LongTensor(targets)
        self.ids = ids

    def __getitem__(self, index):
        return self.static[index], self.targets[index], self.ids[index]

    def __len__(self):
        return self.targets.size(0)

class StaticLoader():
    def __init__(self):
        #read data
        positive_data = pd.read_csv(data_path+"/positive_dataset.csv")
        negative_data = pd.read_csv(data_path+"/negative_dataset.csv")
        positive_data.index = positive_data.ecg_id
        negative_data.index = negative_data.ecg_id

        #label for each type of CVD
        positive_data["label"] = 1
        positive_data.loc[positive_data.CVD_type.values == "CHD","label"] = 1 
        positive_data.loc[positive_data.CVD_type.values == "Stroke","label"] = 2
        positive_data.loc[positive_data.CVD_type.values == "HF","label"] = 3

        #label for negative data(non-CVD)
        negative_data["label"] = 0
        self.labels_dict = {0:"non-CVD", 1:"CHD", 2:"Stroke", 3:"HF"}

        #ecg_id is unique idx for each record
        self.ids = np.concatenate([positive_data.ecg_id.values, negative_data.ecg_id.values])
        self.labels = pd.concat([positive_data["label"],negative_data["label"]]).values
        self.subject_id = pd.concat([positive_data["subject_id"],negative_data["subject_id"]]).values
        self.targets_df = pd.DataFrame({"labels":self.labels, "subject_id":self.subject_id}, index=self.ids)
        
        #features
        self.feats_demographic = ['gender', 'age']
        self.feats_outpatient = ['SBP', 'SBP_flag', 'DBP', 'DBP_flag', 'Weight', 'Weight_flag', 'Height', 'Height_flag']
        self.feats_medical = ['hyperlipidemia', 'DM', 'afib', 'hypertension', 'PAD', 'CHD', 'CABG','PCI', 'Stroke', 'HF']
        self.feats_medication = ['C01A', 'C01B', 'C01C', 'C01D', 'C01E', 'C02A', 'C02C', 'C02D', 'C02K', 'C03A', 'C03B', 'C03C', 'C03D', 'C03E',
                        'C03X', 'C04A', 'C05A', 'C05B', 'C07A', 'C08C', 'C08D', 'C09A', 'C09B', 'C09C', 'C09D', 'C09X', 'C10A']
        
        self.feats_core = self.feats_demographic + self.feats_outpatient 
        self.feats_extended = self.feats_core + self.feats_medical + self.feats_medication
        
        #subset of data
        data = pd.concat([positive_data, negative_data])
        self.data_core = data[self.feats_core].values
        self.data_extended = data[self.feats_extended].values
        
    def __len__(self):
        return len(self.ids)
    
    def get_ids(self):
        return self.ids
    
    def get_data(self):
        return self.data_core, self.data_extended
    
    def get_targets(self):
        return self.targets_df
    
    def get_labels_dict(self):
        return self.labels_dict
    
    def get_dataset(self, type):
        if type == "core":
            return StaticDataset(self.data_core, self.labels, self.ids)
        
        if type == "extended":
            return StaticDataset(self.data_extended, self.labels, self.ids)
        
        return None
    
