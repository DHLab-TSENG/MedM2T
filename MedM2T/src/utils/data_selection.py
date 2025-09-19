from sklearn.model_selection import GroupKFold,GroupShuffleSplit
import numpy as np
import random
import copy

def train_valid_test_kfolds_for_task1(ecg_id_list, labels, subject_id_list, random_seed = 0):
    """
    Split the dataset into train, validation, and test sets by subjects
    Args:
        ecg_id_list (list): List of ECG IDs
        labels (list): List of labels
        subject_id_list (list): List of subject IDs
        random_seed (int): Random seed
    Returns:
        kfolds (list): List of train, validation, and test indices for each fold
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    train_test_kfold = GroupKFold(n_splits=5)
    train_valid_split = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=random_seed)
    kfolds = []
    for train_valid_idx, test_idx in train_test_kfold.split(ecg_id_list, labels, subject_id_list):
        _train_idx, _valid_idx = list(train_valid_split.split(train_valid_idx, labels[train_valid_idx], subject_id_list[train_valid_idx]))[0]
        train_idx, valid_idx = train_valid_idx[_train_idx], train_valid_idx[_valid_idx]
        kfolds.append([train_idx, valid_idx, test_idx])
    return kfolds

def train_valid_test_kfolds_for_task2(subject_id_list, subject_groups, random_seed = 0):
    """
    Split the dataset into train, validation, and test sets by subjects
    Args:
        subject_id_list (list): List of subject IDs
        subject_groups (list): List of subject groups
        random_seed (int): Random seed
    Returns:
        kfolds (list): List of train, validation, and test indices for each fold
    """
    from sklearn.model_selection import KFold, train_test_split
    np.random.seed(random_seed)
    random.seed(random_seed)
    train_test_kfold = KFold(n_splits=5, shuffle = True, random_state=random_seed)
    kfolds = []
    for train_valid_idx, test_idx in train_test_kfold.split(subject_id_list, subject_groups):
        train_valid_groups = subject_groups[train_valid_idx]
        train_idx, valid_idx, train_groups, test_groups = train_test_split(
            train_valid_idx, train_valid_groups, test_size=0.2, shuffle = True, random_state=random_seed
        )
        kfolds.append([train_idx, valid_idx, test_idx])
    return kfolds

def get_sub_kfolds(ids, sub_ids, kfolds):
    sub_ids_dict = {k:v for k, v in zip(sub_ids,range(len(sub_ids)))} 
    sub_kfolds= []
    for train_idx, valid_idx, test_idx in kfolds:
        sub_train_idx = [sub_ids_dict[id] for id in ids[train_idx] if id in sub_ids_dict]
        sub_valid_idx = [sub_ids_dict[id] for id in ids[valid_idx] if id in sub_ids_dict]
        sub_test_idx = [sub_ids_dict[id] for id in ids[test_idx] if id in sub_ids_dict]
        sub_kfolds.append([sub_train_idx, sub_valid_idx, sub_test_idx])
    return sub_kfolds

def get_sub_kfolds_byidx(kfolds, idx):
    """
    Get the sub_kfolds by index
    Args:
        kfolds (list): List of train, validation, and test indices for each fold
        idx (list): List of indices
    Returns:
        sub_kfolds (list): List of train, validation, and test indices for each fold
    """
    sub_kfolds = []
    for train_idx, valid_idx, test_idx in kfolds:
        sub_train_idx = np.intersect1d(train_idx, idx)
        sub_valid_idx = np.intersect1d(valid_idx, idx)
        sub_test_idx = np.intersect1d(test_idx, idx)
        sub_kfolds.append([sub_train_idx, sub_valid_idx, sub_test_idx])
    return sub_kfolds