import numpy as np
import pandas as pd

# Function to get the mean value for each subject in the DataFrame
def get_subject_mean(data_df):
    sub_groups = data_df.groupby("subject_id")
    mean_list = []
    for k,v in sub_groups["value"]:
        mean_list.append(np.nanmean(v.values.astype(float)))
    return np.array(mean_list)

# Function to get the cut bins for numerical items, using percentiles (picewise linear encoding)
def items_cuts(item_groups, s_idx=0, values_bins=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95]):
    cut_bins = {}
    for itemid, _sub_df in item_groups:
        values = get_subject_mean(_sub_df)
        values = values[~np.isnan(values)]
        bins = np.percentile(values, values_bins)
        bins = np.unique(bins)
        bins = np.concatenate([[-np.inf],bins,[np.inf]])
        cut_bins[itemid] = {}
        cut_bins[itemid]["bins"] = bins
        cut_bins[itemid]["labels"] = [i for i in np.arange(0,len(bins)-1)+s_idx]
        s_idx += len(bins)-1
        
    return cut_bins, s_idx

# Function to get the categories and labels for categorical items
def categories_labels(item_groups, s_idx=0):
    item_labels = {}

    for itemid, labs_df in item_groups:
        categories = labs_df['value'].unique()
        item_labels[itemid] = {}
        item_labels[itemid]["categories"] = categories
        item_labels[itemid]["labels"] = [i for i in np.arange(0,len(categories))+s_idx]
        s_idx += len(categories)

    return item_labels, s_idx

# Function to get the cut bins for numerical items, using Gini impurity
def items_cuts_gini(labs_groups, targets_df, labels_cnt=None):
    from sklearn.tree import DecisionTreeClassifier
    def get_bin_ranges(tree):
        thresholds = tree.tree_.threshold
        bins = []

        def traverse(node_id, lower, upper):
            if thresholds[node_id] != -2:  
                threshold = thresholds[node_id]
                left_child = tree.tree_.children_left[node_id]
                right_child = tree.tree_.children_right[node_id]
                traverse(left_child, lower, min(upper, threshold))
                traverse(right_child, max(lower, threshold), upper)
            else: 
                bins.append((lower, upper))

        traverse(0, float("-inf"), float("inf"))
        return sorted(bins)

    cut_bins = {}
    if labels_cnt is None:
        labels_cnt = 0
    for _itemid, _labs_df in labs_groups:
        value = _labs_df.value.values.reshape(-1, 1)
        ecg_id = _labs_df.ecg_id.values
        flag = targets_df.loc[ecg_id].labels.values
        tree = DecisionTreeClassifier(
            criterion="gini",
            max_leaf_nodes=10,
            min_samples_leaf=int(len(value)*0.05)
        )
        tree.fit(value, flag)
        bin_ranges = get_bin_ranges(tree)

        bins = []
        for bin in bin_ranges:
            bins.append(bin[0])
        bins.append(bin[-1])

        cut_bins[_itemid] = {}
        cut_bins[_itemid]["bins"] = bins
        cut_bins[_itemid]["labels"] = [i for i in np.arange(labels_cnt,labels_cnt+len(bins)-1)]
        labels_cnt += len(bins)-1
    
    return cut_bins, labels_cnt

# Function to get the labels dictionary for numerical and categorical items
def get_labels_dict(data_df, numerical_itemid, categorical_itemid = None, num_bins=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95]):
    labels_dict = {}
    s_idx = 0
    if numerical_itemid is not None:
        num_groups = data_df[np.isin(data_df.item, numerical_itemid)].groupby("item")
        num_dict,s_idx = items_cuts(num_groups, s_idx, num_bins)
        for k, v in num_dict.items():
            labels_dict[k] = v
            labels_dict[k]["type"] = "Numeric"

    if categorical_itemid is not None:
        cat_groups = data_df[np.isin(data_df.item, categorical_itemid)].groupby("item")
        cat_dict,s_idx = categories_labels(cat_groups, s_idx)
        for k, v in cat_dict.items():
            labels_dict[k] = v
            labels_dict[k]["type"] = "Category"
    
    None_label = s_idx
    return labels_dict, None_label

# Function to set labels for the original DataFrame based on the labels dictionary
def setting_labels(orig_df, labels_dict):
    source_dict = {itemid:i for i, itemid in enumerate(labels_dict.keys())}
    source_None_label = len(source_dict)

    items_groups = orig_df.groupby("item")
    orig_df["value_label"] = None
    orig_df["source_label"] = None
    for itemid, labs_df in items_groups:
        if itemid not in labels_dict:
            continue
        if labels_dict[itemid]["type"] == "Numeric":
            values = labs_df["value"].values.astype(float)
            labels = pd.cut(values, bins = labels_dict[itemid]["bins"], labels = labels_dict[itemid]["labels"])
        else:
            val2label = {k:v for k,v in zip(labels_dict[itemid]["categories"], labels_dict[itemid]["labels"])}
            values = labs_df["value"].values
            labels = [val2label[v] for v in values]
        orig_df.loc[labs_df.index,"value_label"] = list(labels)
        orig_df.loc[labs_df.index,"source_label"] = source_dict[itemid]

    return orig_df, source_dict, source_None_label

# Function to get the source and value labels for each time window
def source_value_win_bins(groups, value_None_label, source_None_label=None, only_value=False):
    def tolist(x):
        if len(x) == 0:
            return [value_None_label]
        return list(x)

    def source_tolist(x):
        if len(x) == 0:
            return [source_None_label]
        return list(x)

    win_values = {}
    win_sources = {} if not only_value else None
    for id, sub_df in groups:
        _values = sub_df.groupby("time_window")["value_label"].agg(tolist)
        win_values[id] = _values.values
        
        if not only_value:
            _sources = sub_df.groupby("time_window")["source_label"].agg(source_tolist)
            win_sources[id] = _sources.values

    return win_values, win_sources

# Function to aggregate data by time window for numerical items
def aggregate_by_time_nums(data_groups, tw_size, items_list, fillna_type, fill_value=None):
    labs_data_dict = {}
    for id, _tmp_df in data_groups:
        _tmp_groups = _tmp_df.groupby(["time_window", "item"])
        _means = _tmp_groups["value"].mean().reset_index()
        _means = _means.pivot(index="time_window", columns="item", values="value")
        _tmp_df = pd.DataFrame(index=np.arange(0,tw_size), columns=items_list)
        _tmp_df.loc[_means.index, _means.columns] = _means
        if fillna_type == "forward":
            _tmp_df = _tmp_df.fillna(method="ffill")
            _tmp_df = _tmp_df.fillna(value=0)
        elif fillna_type == "zero":
            _tmp_df = _tmp_df.fillna(value=0)
        elif fillna_type == "value":
            _tmp_df = _tmp_df.fillna(value=fill_value)
        labs_data_dict[id] = _tmp_df
    return labs_data_dict

# Function to aggregate data by time window for categorical items
def aggregate_by_time_cats(data_groups, tw_size, items_dict, fillna_type):
    items_list = []
    for k, v in items_dict.items():
        for _v in v:
            items_list.append(str(k)+"_"+str(_v))
    labs_data_dict = {}
    for id, _tmp_df in data_groups:
        _tmp_groups = _tmp_df.groupby(["time_window", "item"])
        _tmp_df = _tmp_df.loc[_tmp_groups["delta_time"].idxmax()]
        _tmp_df["item"] = _tmp_df["item"].astype(str) + "_" + _tmp_df["value"].astype(str)
        _tmp_df["value"] = 1
        _values = _tmp_df.pivot(index='time_window', columns='item', values='value')
        _tmp_df = pd.DataFrame(index=np.arange(0,tw_size), columns=items_list)
        _tmp_df.loc[_values.index, _values.columns] = _values
        if fillna_type == "forward":
            _tmp_df = _tmp_df.fillna(method="ffill")
            _tmp_df = _tmp_df.fillna(value=0)
        elif fillna_type == "zero":
            _tmp_df = _tmp_df.fillna(value=0)
        labs_data_dict[id] = _tmp_df
    return labs_data_dict, items_list 

# Function to get the time bins and time windows for the data, based on the quantile of delta time
def get_time_bins(delta_time, percentiles):
    time_bins = np.percentile(delta_time, percentiles)
    time_bins = np.unique(time_bins)
    time_wins = np.digitize(delta_time, time_bins)
    return time_bins, time_wins

from scipy.signal import find_peaks
# Define a function to compute the statistics for each series
# HAIM
def compute_event_stats(series):
    result = {}
    series = series.dropna()  # Drop NaN values
    
    if len(series) > 0:
        result['max'] = series.max()
        result['min'] = series.min()
        result['mean'] = series.mean(skipna=True)
        result['variance'] = series.var(skipna=True)
        result['meandiff'] = series.diff().mean()  # Average change
        result['meanabsdiff'] = series.diff().abs().mean()
        result['maxdiff'] = series.diff().abs().max()
        result['sumabsdiff'] = series.diff().abs().sum()
        result['diff'] = series.iloc[-1] - series.iloc[0]
        
        # Compute the number of peaks
        peaks, _ = find_peaks(series)
        result['npeaks'] = len(peaks)
        
        # Compute the trend (linear slope)
        if len(series) > 1:
            result['trend'] = np.polyfit(np.arange(len(series)), series, 1)[0]
        else:
            result['trend'] = 0
    else:
        # If there is no data, fill with NaN or default values as needed
        result = {key: np.nan for key in ['max', 'min', 'mean', 'variance', 'meandiff',
                                          'meanabsdiff', 'maxdiff', 'sumabsdiff', 'diff', 
                                          'npeaks', 'trend']}
    
    return result

def get_num_stats(num_groups):
    stats_num_dict = {}
    for subject_id, _data in num_groups:
        _data.sort_values("delta_time", inplace=True)
        _groups = _data.groupby("item")
        stats = _groups["value"].agg(compute_event_stats)
        last_idx = _groups["delta_time"].idxmax()
        _data_last = _data.loc[last_idx]
        _data_last.index = _data_last.item
        _tmp = {}
        for itemid, res in stats.items():
            _tmp[str(itemid)+"_last"] = _data_last.loc[itemid].value
            _tmp[str(itemid)+"_hours"] = _data_last.loc[itemid].delta_time
            _tmp.update({"%s_%s"%(itemid, k):v for k,v in res.items()})
        stats_num_dict[subject_id] = _tmp
    stats_num_df = pd.DataFrame(stats_num_dict).T
    return stats_num_df

def get_cat_stats(cat_groups):
    stats_cat_dict = {}
    for subject_id, _data in cat_groups:
        _data.sort_values("delta_time", inplace=True)
        _groups = _data.groupby("item")
        stats = _groups["value"].value_counts()
        _tmp = {"%s_%s"%(k[0],k[1]):v for k,v in stats.to_dict().items()}
        stats_cat_dict[subject_id] = _tmp
    stats_cat_df = pd.DataFrame(stats_cat_dict).T
    return stats_cat_df