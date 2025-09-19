import pandas as pd
import json
import numpy as np
import os
import re
# import logging

def extract_GFR(comments):
    match_str1 = "Estimated GFR = "
    match_str2 = "between "
    match_str3 = "estimated GFR (eGFR) is likely"
    
    if match_str3 in comments:
        return ">75", ">75"
    
    idx = comments.find(match_str1)
    val1, val2 = None, None
    if idx > 0:
        s = idx+len(match_str1)
        val1 = comments[s:s+5].split(" ")[0]
        comments = comments[s:]
        s = comments.find(match_str1)+len(match_str1)
        val2 = comments[s:s+5].split(" ")[0]
    else:
        idx = comments.find(match_str2)
        if idx > 0:
            s = idx+len(match_str2)
            tmp = comments[s:s+20].split(" ")
            val1 = tmp[0]
            val2 = tmp[2]
        
    return val1, val2  

def labevents_handel(itemid, item_infor, item_excluded, labs_df):
    labs_df["parse_value"] = None
    
    if item_excluded is not None:
        idx_excluded = labs_df.comments.isin(item_excluded)
        labs_df = labs_df[~idx_excluded]
        print("Exclude %s records"%(sum(idx_excluded)))
        
    if itemid == 50920: #eGFR
        comments_idx = (labs_df["parse_value"].isna()) & (labs_df["comments"].notna()) 
        eGFR_value = []
        for x in labs_df.loc[comments_idx,"comments"].values:
            v1, v2 = extract_GFR(x)
            if v1 is None and v2 is None:
                eGFR_value.append(None)
            else:
                v1 = float(v1.replace(">",""))
                v2 = float(v2.replace(">",""))
                item_val = (float(v1)+float(v2))/2
                eGFR_value.append(item_val)
        labs_df.loc[comments_idx,"parse_value"] = eGFR_value
        return labs_df
        
    #item type is numeric
    if item_infor["type"] == "num":
        labs_df["parse_value"] = labs_df["valuenum"].values
        
        #convert values(str) to numeric 
        value_idx = (labs_df["parse_value"].isna()) & (labs_df["value"].notna()) 
        value2valuenum = [] 
        for x in labs_df[value_idx]["value"]:
            x = re.sub('[<>=]', "" ,x) 
            if re.sub('[.]', "" ,x, 1).isnumeric():
                value2valuenum.append(float(x))
            elif re.search("[~-]", x): #value range ex:10-20
                tmp = [float(v) for v in re.split('[~-]', x) if re.sub('[.]', "" ,v, 1).isnumeric()]
                if len(tmp) > 0:
                    value2valuenum.append(np.mean(tmp))
                else:
                    value2valuenum.append(None)
            else:
                value2valuenum.append(None)
                
        labs_df.loc[value_idx,"parse_value"] = value2valuenum
        
        #convert comments to numeric
        if "comments" in item_infor:
            comments_idx = (labs_df["parse_value"].isna()) & (labs_df["comments"].notna()) 
            comments2valuenum = [item_infor["comments"].get(x) if item_infor["comments"].get(x) else None for x in labs_df[comments_idx]["comments"]]
            labs_df.loc[comments_idx,"parse_value"] = comments2valuenum
            
        labs_df["parse_value"] = labs_df["parse_value"].astype(float)
    
    #item type is category
    else:
        category = item_infor["category"]
        value_idx = (labs_df["parse_value"].isna()) & (labs_df["value"].isin(category)) 
        labs_df.loc[value_idx,"parse_value"] = labs_df.loc[value_idx,"value"]
        comments_idx = (labs_df["parse_value"].isna()) & (labs_df["comments"].isin(category)) 
        labs_df.loc[comments_idx,"parse_value"] = labs_df.loc[comments_idx,"comments"]

        if "mapping" in item_infor:
            mapping = item_infor["mapping"]
            value_idx = (labs_df["parse_value"].isna()) & (labs_df["value"].isin(mapping.keys())) 
            labs_df.loc[value_idx,"parse_value"] = [mapping.get(x) for x in labs_df.loc[value_idx,"value"]]
            comments_idx = (labs_df["parse_value"].isna()) & (labs_df["comments"].isin(mapping.keys())) 
            labs_df.loc[comments_idx,"parse_value"] = [mapping.get(x) for x in labs_df.loc[comments_idx,"comments"]]
        
        if "value2category" in item_infor:
            value2category = item_infor["value2category"]
            keys = list(value2category.keys())
            intervals = [tuple(x) for x in value2category.values()]
            bins = pd.IntervalIndex.from_tuples(intervals , closed = "left")
            
            valuenum_idx = (labs_df["parse_value"].isna()) & (labs_df["valuenum"]) 
            tmp = pd.cut(labs_df.loc[valuenum_idx, "valuenum"].values, bins=bins)
            to_keys = {interval: key for interval, key in zip(tmp.categories, keys)}
            labs_df.loc[valuenum_idx, "parse_value"] = [to_keys.get(x) for x in tmp]
            
            value_idx = (labs_df["parse_value"].isna()) & (labs_df["value"].str.replace("[.]",'',regex=True).str.isnumeric()) 
            tmp = pd.cut(labs_df.loc[value_idx, "value"].values.astype(float), bins=bins)
            to_keys = {interval: key for interval, key in zip(tmp.categories, keys)}
            labs_df.loc[value_idx, "parse_value"] = [to_keys.get(x) for x in tmp]   
    
    return labs_df

def Ectopy_encode(Ectopy_Type_df, Ectopy_Frequency_df):
    Ectopy_Type_df = Ectopy_Type_df[['subject_id', 'hadm_id','charttime','warning','value']]
    Ectopy_Type_df.columns = ['subject_id', 'hadm_id','charttime','warning','type']
    Ectopy_Type_df = Ectopy_Type_df[Ectopy_Type_df.type.notna()]
    Ectopy_Type_df = Ectopy_Type_df[Ectopy_Type_df.type != "None"]
    Ectopy_Type_df = Ectopy_Type_df.drop_duplicates()
    Ectopy_Type = Ectopy_Type_df.type.unique()
    
    Ectopy_Frequency_df = Ectopy_Frequency_df[['subject_id', 'hadm_id','charttime','warning','value']]
    Ectopy_Frequency_df.columns = ['subject_id', 'hadm_id','charttime','warning','frequency']
    Ectopy_Frequency_df = Ectopy_Frequency_df[Ectopy_Frequency_df.frequency != "None"]
    Ectopy_Frequency_df = Ectopy_Frequency_df.drop_duplicates()
    
    Ectopy_df = Ectopy_Type_df.merge(Ectopy_Frequency_df, on = ['subject_id','hadm_id','charttime'], how = "left") 
    
    freq_to_level = {'Rare':1,'Occasional':2,'Frequent':3, 'Runs Vtach':4}
    level_to_freq = {v:k for k,v in freq_to_level.items()}
    
    Ectopy_groups = Ectopy_df.groupby(by = ['subject_id','hadm_id','charttime','type'])
    data = {x:[] for x in Ectopy_Type}
    for val, sub_df in Ectopy_groups:
        warning = np.nanmax([sub_df['warning_y'].dropna().max(), sub_df['warning_x'].dropna().max()])
        frequency_list = [freq_to_level[x] for x in sub_df['frequency'].dropna().values]
        if len(frequency_list) == 0:
            frequency = "Other"
        else:
            frequency = level_to_freq[max(frequency_list)]
        tmp = {k:v for k,v in zip(['subject_id', 'hadm_id', 'charttime'],val[:-1])}
        tmp["value"] = frequency
        tmp["warning"] = warning
        data[val[-1]].append(tmp)
    
    for Ectopy in Ectopy_Type:
        data[Ectopy] = pd.DataFrame(data[Ectopy])
    
    for Ectopy in Ectopy_Type:
        new_Ectopy = re.sub("[.\']", "", Ectopy)
        new_Ectopy = re.sub("[\/]", "_", new_Ectopy)
        new_Ectopy = re.sub(" ", "_", new_Ectopy)
        new_Ectopy = "Ectopy_%s"%(new_Ectopy)
        
        data[new_Ectopy] = data.pop(Ectopy)
    
    return data

class events_preprocessing():
    def __init__(self):
        self.mimiciv_csv_path = "./mimic_csv"
        self.data_path = "./mimic_data"
        self.labs_fname = "labs"
        self.vitalsigns_fname = "vitalsigns"
        self.chartlabs_fname = "chartlabs"
        self.respiratory_fname = "respiratory"
        self.chartevents = None
        self.labsevents = None
        
    def load(self):
        self.d_labitems = pd.read_csv("%s/d_labitems.csv"%(self.mimiciv_csv_path))
        self.d_items = pd.read_csv("%s/d_items.csv"%(self.mimiciv_csv_path))
        self.first_icu_df = pd.read_csv("%s/first_icu.csv"%(self.data_path))
        
        self.items_df = pd.read_csv("%s/items.csv"%(self.data_path))
        self.chartlabs_items = self.items_df[(self.items_df.table == "chartevents")&(self.items_df.category == "laboratory")]
        self.labs_items = self.items_df[(self.items_df.table == "labevents")&(self.items_df.category == "laboratory")]
        self.vitalsigns_items = self.items_df[(self.items_df.category == "vitalsigns")]
        self.respiratory_items = self.items_df[(self.items_df.category == "respiratory")]
                
        if self.check_exists(self.labs_items, self.labs_fname) == False:
            print("miss item in labs")
            if self.labsevents is None:
                labs_groups = self.load_labevents()
            self.save_items_to_pickle(labs_groups, self.labs_fname)
            
        print("loading labs")
        self.labs_dict = self.load_items(self.labs_items, self.labs_fname)
        
        if self.check_exists(self.vitalsigns_items, self.vitalsigns_fname) == False:
            print("miss item in vitalsigns")
            vitalsigns_groups = self.load_chartevents(self.vitalsigns_fname, "Routine Vital Signs")
            self.save_items_to_pickle(vitalsigns_groups, self.vitalsigns_fname, tbl = "chartevents")
        
        print("loading vitalsigns")
        self.vitalsigns_dict = self.load_items(self.vitalsigns_items, self.vitalsigns_fname)
        
        if self.check_exists(self.chartlabs_items, self.chartlabs_fname) == False:
            print("miss item in chartlabs")
            chartlabs_groups = self.load_chartevents(self.chartlabs_fname, "Labs")
            self.save_items_to_pickle(chartlabs_groups, self.chartlabs_fname, tbl = "chartevents")
        
        print("loading chartlabs")
        self.chartlabs_dict = self.load_items(self.chartlabs_items, self.chartlabs_fname)
        
        if self.check_exists(self.respiratory_items, self.respiratory_fname) == False:
            print("miss item in respiratory")
            respiratory_groups = self.load_chartevents(self.respiratory_fname, "Respiratory")
            self.save_items_to_pickle(respiratory_groups, self.respiratory_fname, tbl = "chartevents")
            
        print("loading respiratory")
        self.respiratory_dict = self.load_items(self.respiratory_items, self.respiratory_fname)
        
        self.chartevents = None
        self.labsevents = None
        
        
    def processing(self):
        
        #labs(labevents) processing
        with open("%s/labevents_value_mapping.json"%(self.data_path), "r") as f:
            self.labevents_value_mapping = json.loads(f.read())
            
        with open("%s/labevents_excluded.json"%(self.data_path), "r") as f:
            self.labevents_excluded = json.loads(f.read())
            
        for itemid, lab_df in self.labs_dict.items():
            item_infor = self.labevents_value_mapping[str(itemid)]
            item_excluded = None
            if str(itemid) in self.labevents_excluded:
                item_excluded = self.labevents_excluded[str(itemid)]
                            
            print("labevents item-%s: %d records"%(itemid, len(lab_df)))
            lab_df = labevents_handel(itemid, item_infor, item_excluded, lab_df.copy())
            idx_na = lab_df.parse_value.isna()
            if any(idx_na):
                lab_df = lab_df[~idx_na]
                print("Drop %s records with missing value"%(sum(idx_na)))
            self.labs_dict[itemid] = lab_df
            
        #merge laboratory items
        self.laboratory = {}
        self.laboratory_itemids = {}
        labs_merge_group = self.labs_items[['itemid_merge', 'itemid']].groupby(by = "itemid_merge")
        for itemid_merge, sub_df in labs_merge_group:
            merge_df = pd.DataFrame()
            self.laboratory_itemids[itemid_merge] = []
            for idx,row in sub_df.iterrows():
                lab_df = self.labs_dict[row.itemid][['subject_id','hadm_id','charttime','flag','parse_value']].copy()
                merge_df = pd.concat([merge_df,lab_df])   
                self.laboratory_itemids[itemid_merge].append(row.itemid)
                print("merge item-%d into item-%d"%(row.itemid, itemid_merge))
            merge_df.columns = ['subject_id', 'hadm_id','charttime','flag','value']
            
            idx_na = merge_df.value.isna()
            if any(idx_na):
                merge_df = merge_df[~idx_na]
                print("Drop %s records with missing value in item-%d"%(sum(idx_na), itemid_merge))
            
            idx_duplicated = merge_df.duplicated()
            if any(idx_duplicated):
                merge_df = merge_df[~idx_duplicated]
                print("Drop %s duplicated records in item-%d"%(sum(idx_duplicated), itemid_merge))
            
            self.laboratory[itemid_merge] = merge_df
        
        chartlabs_merge_group = self.chartlabs_items[['itemid_merge', 'itemid']].groupby(by = "itemid_merge")
        for itemid_merge, sub_df in chartlabs_merge_group:
            merge_df = pd.DataFrame()
            if itemid_merge not in self.laboratory.keys():
                self.laboratory_itemids[itemid_merge] = []
                
            for idx,row in sub_df.iterrows():
                chartlab_df = self.chartlabs_dict[row.itemid][['subject_id', 'hadm_id','charttime','warning','valuenum']].copy()
                merge_df = pd.concat([merge_df,chartlab_df])
                self.laboratory_itemids[itemid_merge].append(row.itemid)
                print("merge item-%d into item-%d"%(row.itemid, itemid_merge))
            
            merge_df.columns = ['subject_id', 'hadm_id','charttime','warning','value']
            
            idx_na = merge_df.value.isna()
            if any(idx_na):
                merge_df = merge_df[~idx_na]
                print("Drop %s records with missing value in item-%d"%(sum(idx_na), itemid_merge))

            idx_duplicated = merge_df.duplicated()
            if any(idx_duplicated):
                merge_df = merge_df[~idx_duplicated]
                print("Drop %s duplicated records in item-%d"%(sum(idx_duplicated), itemid_merge))
            
            if itemid_merge in self.laboratory.keys():
                lab_df = self.laboratory[itemid_merge]
                labs_chartlabs_df = lab_df.merge(merge_df, on = ['subject_id','hadm_id','charttime','value'], how = "outer")
                print("merge chartlabs into labs item-%d"%(itemid_merge))
                self.laboratory[itemid_merge] = labs_chartlabs_df
            else:
                self.laboratory[itemid_merge] = merge_df
                
        #merge vital signs items
        self.vitalsigns = {}
        self.vitalsigns_itemids = {}
        vitalsigns_merge_group = self.vitalsigns_items[['itemid_merge', 'itemid']].groupby(by = "itemid_merge")
        for itemid_merge, sub_df in vitalsigns_merge_group:
            param_type = self.d_items[self.d_items.itemid == itemid_merge].param_type.values[0]
            merge_df = pd.DataFrame()
            self.vitalsigns_itemids[itemid_merge] = []
            for idx,row in sub_df.iterrows():
                if param_type == "Numeric":
                    vitalsign_df = self.vitalsigns_dict[row.itemid][['subject_id','hadm_id','charttime','warning','valuenum']].copy()
                    if row.itemid == 223761: 
                        vitalsign_df['valuenum'] = (vitalsign_df['valuenum']-32)/1.8 # °C = (°F - 32) ÷ 1.8
                else:
                    vitalsign_df = self.vitalsigns_dict[row.itemid][['subject_id','hadm_id','charttime','warning','value']].copy()      
                merge_df = pd.concat([merge_df,vitalsign_df])   
                self.vitalsigns_itemids[itemid_merge].append(row.itemid)
                print("merge item-%d into item-%d"%(row.itemid, itemid_merge))
            merge_df.columns = ['subject_id', 'hadm_id','charttime','warning','value']
            
            idx_na = merge_df.value.isna()
            if any(idx_na):
                merge_df = merge_df[~idx_na]
                print("Drop %s records with missing value in item-%d"%(sum(idx_na), itemid_merge))
            
            idx_duplicated = merge_df.duplicated()
            if any(idx_duplicated):
                merge_df = merge_df[~idx_duplicated]
                print("Drop %s duplicated records in item-%d"%(sum(idx_duplicated), itemid_merge))
            
            self.vitalsigns[itemid_merge] = merge_df

        respiratory_merge_group = self.respiratory_items[['itemid_merge', 'itemid']].groupby(by = "itemid_merge")
        for itemid_merge, sub_df in respiratory_merge_group:
            param_type = self.d_items[self.d_items.itemid == itemid_merge].param_type.values[0]
            merge_df = pd.DataFrame()
            self.vitalsigns_itemids[itemid_merge] = []
            for idx,row in sub_df.iterrows():
                if param_type == "Numeric":
                    respiratory_df = self.respiratory_dict[row.itemid][['subject_id','hadm_id','charttime','warning','valuenum']].copy()
                else:
                    respiratory_df = self.respiratory_dict[row.itemid][['subject_id','hadm_id','charttime','warning','value']].copy()
                merge_df = pd.concat([merge_df,respiratory_df])   
                self.vitalsigns_itemids[itemid_merge].append(row.itemid)
                print("merge item-%d into item-%d"%(row.itemid, itemid_merge))
            merge_df.columns = ['subject_id', 'hadm_id','charttime','warning','value']
            
            idx_na = merge_df.value.isna()
            if any(idx_na):
                merge_df = merge_df[~idx_na]
                print("Drop %s records with missing value in item-%d"%(sum(idx_na), itemid_merge))
            
            idx_duplicated = merge_df.duplicated()
            if any(idx_duplicated):
                merge_df = merge_df[~idx_duplicated]
                print("Drop %s duplicated records in item-%d"%(sum(idx_duplicated), itemid_merge))
            
            self.vitalsigns[itemid_merge] = merge_df
            
        #Ectopy processing
        print("encode vitalsigns ectopy, item-224650,224651")
        Ectopy_Type_df = self.vitalsigns.pop(224650)
        Ectopy_Frequency_df = self.vitalsigns.pop(224651)
        self.Ectopy = Ectopy_encode(Ectopy_Type_df, Ectopy_Frequency_df)
        
    def export(self):
        laboratory_data = pd.DataFrame()
        item_info_list = []
        for itemid, data in self.laboratory.items():
            combine_itemid = self.laboratory_itemids[itemid]
            specimen = None
            category = self.items_df[self.items_df.itemid == itemid].category.iloc[0]
            if itemid in self.labs_items.itemid.values:
                info = self.d_labitems[self.d_labitems.itemid == itemid].iloc[0]
                specimen = info.fluid
                item_type = "Numeric" if self.labevents_value_mapping[str(itemid)]["type"] == "num" else "Text"
                label = info.label
                units = self.labs_dict[itemid].valueuom.unique()[0]
                linksto = "labevents"
            else:
                info = self.d_items[self.d_items.itemid == itemid].iloc[0]
                units = info.unitname
                item_type = info.param_type
                label = info.label
                linksto = "chartevents"
               
            item_info = {"category":category, "itemid":itemid, "linksto":linksto, "label":label, "specimen":specimen, "type":item_type, "units":units, "combine_itemid": combine_itemid}
            item_info["records_num"] = len(data)
            item_info["subjects_num"] = len(data.subject_id.unique())
               
            if item_type == "Text":
                categories_counts = data.value.value_counts()
                categories_counts_str = ""
                for category_item, cnt in zip(categories_counts.index.tolist(), categories_counts.values.tolist()):
                    categories_counts_str += "%s(%d)/"%(category_item, cnt)
                item_info["categories_counts"] = categories_counts_str
            else:
                item_info["value_mean"] = data.value.mean()
                item_info["value_min"] = data.value.min()
                item_info["value_max"] = data.value.max()
                quantile = data.value.quantile([0.25, 0.5, 0.75])
                quantile.index = ["value_25th", "value_50th", "value_75th"]
                item_info.update(dict(quantile))
                
            assert(any(data.value.isna()) == False)  
            data["itemid"] = itemid
            laboratory_data = pd.concat([laboratory_data,data]) 
            item_info_list.append(item_info)
        
        vitalsigns_data = pd.DataFrame()
        for itemid, data in self.vitalsigns.items():
            combine_itemid = self.vitalsigns_itemids[itemid]
            category = self.items_df[self.items_df.itemid == itemid].category.iloc[0]
            info = self.d_items[self.d_items.itemid == itemid].iloc[0]
            units = info.unitname
            item_type = info.param_type
            label = info.label
            linksto = "chartevents"
               
            item_info = {"category":category, "itemid":itemid, "linksto":linksto, "label":label, "type":item_type, "units":units, "combine_itemid": combine_itemid}
            item_info["records_num"] = len(data)
            item_info["subjects_num"] = len(data.subject_id.unique())
               
            if item_type == "Text":
                categories_counts = data.value.value_counts()
                categories_counts_str = ""
                for category_item, cnt in zip(categories_counts.index.tolist(), categories_counts.values.tolist()):
                    categories_counts_str += "%s(%d)/"%(category_item, cnt)
                item_info["categories_counts"] = categories_counts_str
            else:
                item_info["value_mean"] = data.value.mean()
                item_info["value_min"] = data.value.min()
                item_info["value_max"] = data.value.max()
                quantile = data.value.quantile([0.25, 0.5, 0.75])
                quantile.index = ["value_25th", "value_50th", "value_75th"]
                item_info.update(dict(quantile))
                
            assert(any(data.value.isna()) == False)  
            data["itemid"] = itemid
            vitalsigns_data = pd.concat([vitalsigns_data,data]) 
            item_info_list.append(item_info)
        
        Ectopy_item_idx = 0
        for label, data in self.Ectopy.items():
            itemid = "%d_%d"%(224650, Ectopy_item_idx) 
            combine_itemid = [224650,226479,224651,226480]
            category = 'vitalsigns'
            item_type = "Text"
            linksto = "chartevents"
               
            item_info = {"category":category, "itemid":itemid, "linksto":linksto, "label":label, "type":item_type, "combine_itemid": combine_itemid}
            item_info["records_num"] = len(data)
            item_info["subjects_num"] = len(data.subject_id.unique())
               
            categories_counts = data.value.value_counts()
            categories_counts_str = ""
            for category_item, cnt in zip(categories_counts.index.tolist(), categories_counts.values.tolist()):
                categories_counts_str += "%s(%d)/"%(category_item, cnt)
            item_info["categories_counts"] = categories_counts_str
                
            assert(any(data.value.isna()) == False)  
            data["itemid"] = itemid
            vitalsigns_data = pd.concat([vitalsigns_data,data]) 
            item_info_list.append(item_info)
            Ectopy_item_idx+=1
            
        item_info_df = pd.DataFrame(item_info_list)
        
        subjects_num = len(self.first_icu_df)
        item_info_df["subjects_num(%)"] = item_info_df["subjects_num"]/subjects_num*100
        print("export %s/items_info.csv"%(self.data_path))
        item_info_df.to_csv("%s/items_info.csv"%(self.data_path), index = False)
        
        print("export %s/laboratory_measurements.csv"%(self.data_path))
        laboratory_data.to_csv("%s/laboratory_measurements.csv"%(self.data_path), index = False)
        print("export %s/vitalsigns.csv"%(self.data_path))
        vitalsigns_data.to_csv("%s/vitalsigns.csv"%(self.data_path), index = False)
        
        
    def save_items_to_pickle(self, item_groups, out_dir, tbl = "labevents"):
        dir_path = "%s/%s"%(self.data_path,out_dir)
        if os.path.exists(dir_path) == False:
            os.mkdir(dir_path)
            
        for itemid in item_groups.groups.keys():
            out_path = "%s/%s/%d.pkl"%(self.data_path,out_dir,itemid)
            if os.path.exists(out_path):
                continue
            else:
                item_df = item_groups.get_group(itemid)
                if tbl == "labevents":
                    item_df = item_df.merge(self.first_icu_df[["hadm_id","icu_intime","icu_outtime"]], how = "inner", on = "hadm_id")
                    valid_idx = (item_df["charttime"] >= item_df["icu_intime"]) &  (item_df["charttime"] <= item_df["icu_outtime"])
                    item_df = item_df[valid_idx]
                else: #tbl is chartevents
                    item_df = item_df[item_df.stay_id.isin(self.first_icu_df.stay_id)]
                
                print("saving %s"%(out_path))
                item_df.to_pickle(out_path)
    
    def get_chartevents_groups(self, fname, category):
        if os.path.exists("%s/%s/%s.pkl"%(self.data_path, fname, fname)):
            print("loading chartevents-%s"%(category))
            items = pd.read_pickle("%s/%s/%s.pkl"%(self.data_path, fname, fname))
        else:
            if self.chartevents is None:
                print("loading chartevents")
                self.chartevents = pd.read_csv("%s/chartevents.csv"%(self.mimiciv_csv_path))
            itemid = self.d_items.itemid[self.d_items.category == category].values
            items = self.chartevents[self.chartevents.itemid.isin(itemid)]
            if os.path.exists("%s/%s"%(self.data_path,fname)) == False:
                os.mkdir("%s/%s"%(self.data_path,fname))
            print("saving %s"%("%s/%s/%s.pkl"%(self.data_path, fname, fname)))
            items.to_pickle("%s/%s/%s.pkl"%(self.data_path, fname, fname))
        item_groups = items.groupby(by = "itemid")
        return item_groups
        
    def load_chartevents(self, fname, category):
        return self.get_chartevents_groups(fname,category)
                
    def load_labevents(self):
        print("loading labsevents")
        if os.path.exists("%s/%s/%s.pkl"%(self.data_path, self.labs_fname, self.labs_fname)):
            labevents = pd.read_pickle("%s/%s/%s.pkl"%(self.data_path, self.labs_fname, self.labs_fname))
        else:
            labevents =  pd.read_csv("%s/labevents.csv"%(self.mimiciv_csv_path))
            if os.path.exists("%s/%s"%(self.data_path,self.labs_fname)) == False:
                os.mkdir("%s/%s"%(self.data_path,self.labs_fname))
            labevents.to_pickle("%s/%s/%s.pkl"%(self.data_path, self.labs_fname, self.labs_fname))
        labs_groups = labevents.groupby(by = "itemid")
        return labs_groups
    
    def check_exists(self, items, items_dir):
        for itemid in items.itemid: 
            if os.path.exists("%s/%s/%d.pkl"%(self.data_path, items_dir, itemid)) == False:
                return False
        return True
    
    def load_items(self, items, items_dir):
        items_dict = {}
        for itemid in items.itemid: 
            item = pd.read_pickle("%s/%s/%d.pkl"%(self.data_path,items_dir, itemid)) 
            items_dict[itemid] = item
                
        return items_dict
    
if __name__ == "__main__": 
    process = events_preprocessing()
    print("=====loading=====")
    process.load()
    print("=====processing=====")
    process.processing()
    print("=====exporting=====")
    process.export() 
