from .config import data_path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
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
        ##Read Patients
        patients = pd.read_csv(data_path+"/patients.csv")
        patients.admittime = pd.to_datetime(patients.admittime)
        patients.dischtime = pd.to_datetime(patients.dischtime)
        patients.icu_intime = pd.to_datetime(patients.icu_intime)
        patients.icu_outtime = pd.to_datetime(patients.icu_outtime)
        patients.deathtime = pd.to_datetime(patients.deathtime)
        patients["gender"] = [0 if x == "F" else 1 for x in patients['gender'].values]
        death_timedelta = (patients.deathtime - patients.icu_intime)
        death_days = [x.total_seconds()/(60*60*24) if pd.notnull(x) else x for x in death_timedelta]
        patients["death_days"] = death_days
        patients.index = patients.subject_id
        self.icu24_patients = patients[patients.icustay_days >= 1]

        self.feats_cols = ["gender", "anchor_age"]
        adm_type_encoder = OneHotEncoder().fit(self.icu24_patients[["admission_type"]].values)
        adm_type = adm_type_encoder.transform(self.icu24_patients[["admission_type"]].values).toarray()
        self.feats_cols += list(adm_type_encoder.categories_[0])

        adm_loc_encoder = OneHotEncoder().fit(self.icu24_patients[["admission_location"]].values)
        adm_loc = adm_loc_encoder.transform(self.icu24_patients[["admission_location"]].values).toarray()
        self.feats_cols += list(adm_loc_encoder.categories_[0])
        
        self.data = np.concatenate((self.icu24_patients[["gender", "anchor_age"]].values, adm_type, adm_loc), axis=1)

        self.labels_dict = {0:"alive", 1:"dead"}
        self.labels =  self.icu24_patients.hospital_expire_flag.values.astype(int)
        self.ids = self.icu24_patients.index.values
        self.targets_df = pd.DataFrame({"labels":self.labels}, index=self.ids)
        
        CICU_list = ["Cardiac Vascular Intensive Care Unit (CVICU)", "Coronary Care Unit (CCU)"]
        self.CICU_idx = np.where(np.isin(self.icu24_patients.first_careunit.values, CICU_list))[0]
        self.CICU_ids = self.ids[self.CICU_idx]
        
    def __len__(self, CICU=False):
        if CICU:
            return len(self.CICU_ids)
        return len(self.ids)
    
    def get_ids(self, CICU=False):
        if CICU:
            return self.CICU_ids
        return self.ids
    
    def get_data(self, CICU=False):
        if CICU:
            return self.data[self.CICU_idx]
        return self.data
    
    def get_targets(self, CICU=False):
        if CICU:
            return self.targets_df.loc[self.CICU_ids]
        return self.targets_df
    
    def get_labels_dict(self):
        return self.labels_dict

    def get_dataset(self, CICU=False):
        if CICU:
            return StaticDataset(self.data[self.CICU_idx], self.labels[self.CICU_idx], self.CICU_ids)
        return StaticDataset(self.data, self.labels, self.ids)
