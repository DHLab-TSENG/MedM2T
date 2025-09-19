import traceback
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import numpy as np 
import copy
import random
import time
from config import device, log_folder
from datetime import datetime
import json
import os
import gc
from torch.nn import functional as F

def set_seed(seed = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
#    torch.use_deterministic_algorithms = True
#    torch.backends.cudnn.benchmark = False

import random
def shuffle(array, random_seed = 0):
    np.random.seed(random_seed)
    random.seed(random_seed)
    sampler_array = copy.deepcopy(array)
    np.random.shuffle(sampler_array)
    return sampler_array

def multiclass_auroc_auprc(y_true, y_probs):
    _y_true = np.array(y_true)
    n_classes = y_probs.shape[1]
    auprcs = []
    aurocs = []
    for i in range(n_classes):
        y_true_binary = (_y_true == i).astype(int) 
        #auroc
        fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, i])
        auroc = auc(fpr, tpr)
        aurocs.append(auroc)
        #auprc
        precision, recall, _ = precision_recall_curve(y_true_binary, y_probs[:, i])
        auprc = auc(recall, precision)
        auprcs.append(auprc)

    class_counts = np.bincount(y_true)
    # weighted_auprc = np.average(auprcs, weights=class_counts)
    # return weighted_auprc
    macro_auprc = np.average(auprcs)
    macro_auroc = np.average(aurocs)
    return macro_auroc, aurocs, macro_auprc, auprcs, class_counts

def train(model, dataloader, optimizer, criterion, epoch, to_device = True, classification = True):
    model.train()
    total_loss, total_count = 0, 0
    for idx, batch in enumerate(dataloader):
        if len(batch[-1]) < 2: # skip batch if batch size is less than 2, which causes error in batch normalization
            continue
        if len(batch) == 3:
            input_data = batch[0].to(device) if to_device else batch[0]
        else:
            if to_device:
                input_data = [x.to(device) for x in batch[:-2]]
            else:
                input_data = [x for x in batch[:-2]]
        targets = batch[-2].to(device) if to_device else batch[-2]
        # index = batch[-1]
        predited = model(input_data)
        if classification == False:
            predited = predited.view(-1)
        loss = criterion(predited, targets)/targets.size(0)
        total_loss += loss*targets.size(0)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        total_count += targets.size(0)
    print('| epoch {:3d} | {:5d}/{:5d} batches '
          '| loss {:8.5f}'.format(epoch, idx, len(dataloader), 
                                  total_loss/total_count))
    return total_loss/total_count

def evaluate(model, dataloader, criterion, to_device = True, classification = True):
    model.eval()
    total_loss, total_count = 0, 0
    pred_list = []
    y_list = []
    subject_id_list = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if len(batch) == 3:
                input_data = batch[0].to(device) if to_device else batch[0]
            else:
                if to_device:
                    input_data = [x.to(device) for x in batch[:-2]]
                else:
                    input_data = [x for x in batch[:-2]]
            targets = batch[-2].to(device) if to_device else batch[-2]
            subject_ids = batch[-1]
            # index = batch[-1]
            predited = model(input_data)
            if classification == False:
                predited = predited.view(-1)
            loss = criterion(predited, targets)/targets.size(0)
            total_loss += loss*targets.size(0)
            total_count += targets.size(0)
            pred_list.extend(predited.cpu().detach().numpy())
            y_list.extend(targets.cpu().detach().numpy())
            subject_id_list.extend([int(x.item()) for x in subject_ids])
    
    pred_list = np.array(pred_list)
    y_list = np.array(y_list)
    if classification:
        # prob_list = np.array([np.exp(x.numpy())/sum(np.exp(x.numpy())) for x in pred_list])
        prob_list = F.softmax(torch.tensor(pred_list), dim = 1)
        prob_list = prob_list.numpy()
        # y_list = np.array([x.numpy() for x in y_list])
        if prob_list.shape[1] > 2: # multiclass
            auroc, _, auprc, _, _ = multiclass_auroc_auprc(y_list, prob_list)
        else: # binary class
            auroc = roc_auc_score(y_list, prob_list[:,1])
            precision, recall, thresholds = precision_recall_curve(y_list, prob_list[:,1])
            auprc = auc(recall, precision)
        return total_loss/total_count, auroc, auprc, prob_list, y_list, subject_id_list
    else:
        # pred_list = np.array([x.numpy() for x in pred_list])
        # y_list = np.array([x.item() for x in y_list])
        return total_loss/total_count, pred_list, y_list, subject_id_list

def run(model, model_name, dataset, batch_size, train_idx, val_idx, test_idx, lr, max_epochs, 
        optimizer_type = "SGD", scheduler_config = None , 
        early_stop_thr = 0.001, early_cnt_thr = 10, 
        loss_weight=None, to_device = True, collate_fun = None, 
        random_seed = 0, classification = True):
    
    sampler_g = torch.Generator().manual_seed(random_seed)
    if collate_fun is None:
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx, generator=sampler_g), pin_memory=True)
        valid_dataloader =  DataLoader(dataset, batch_size=batch_size, sampler=val_idx, pin_memory=True)
        test_dataloader =  DataLoader(dataset, batch_size=batch_size, sampler=test_idx, pin_memory=True)
    else:
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx, generator=sampler_g), collate_fn=collate_fun)
        valid_dataloader =  DataLoader(dataset, batch_size=batch_size, sampler=val_idx, collate_fn = collate_fun)
        test_dataloader =  DataLoader(dataset, batch_size=batch_size, sampler=test_idx, collate_fn = collate_fun)
    
    if classification:
        if loss_weight is not None:
            criterion = torch.nn.NLLLoss(weight=loss_weight, reduction='sum')
        else:
            criterion = torch.nn.NLLLoss(reduction='sum')
    else:
        criterion = torch.nn.MSELoss(reduction='sum')
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
    if scheduler_config is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config["step_size"], gamma=scheduler_config["gamma"])
    
    min_val_loss = np.inf
    best_model = None
    val_epoch_loss = []
    train_epoch_loss = []
    
    stop_cnt = 0
    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()
        loss_train = train(model, train_dataloader, optimizer, criterion, epoch, to_device, classification)
        loss_train = loss_train.detach().item()
        train_epoch_loss.append(loss_train)
        if classification:
            loss_val, auroc_val, auprc_val, _, _, _ = evaluate(model, valid_dataloader, criterion, to_device)
        else:
            loss_val, _, _, _ = evaluate(model, valid_dataloader, criterion, to_device, classification)
        loss_val = loss_val.item()
        val_epoch_loss.append(loss_val)
        if loss_val < min_val_loss:
            min_val_loss = loss_val
            best_model = copy.deepcopy(model)
        
        if scheduler_config is not None:
            scheduler.step()
        
        print('-' * 59)
        if classification:
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                    'valid loss {:8.5f} | valid auroc {:8.3f} | valid auprc {:8.3f}'.format(epoch,
                                                    time.time() - epoch_start_time,
                                                    loss_val, auroc_val, auprc_val))
        else:
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:8.5f}'.format(
                epoch, time.time() - epoch_start_time, loss_val))
            
        print('-' * 80)
        
        if ((loss_val - min_val_loss) > early_stop_thr and (loss_val - loss_train) > 0.0005) or (loss_val - min_val_loss) > early_stop_thr*3:
            stop_cnt += 1
            if stop_cnt > early_cnt_thr:
                print("early stopping in epoch ", epoch)
                break
            else:
                print("early stop cnts", stop_cnt)
        else:
            stop_cnt = max(0,stop_cnt-1)

    print('Checking the results of test dataset.')
    if classification:
        loss_test, auroc_test, auprc_test, prob_list, y_list, subject_id_list = evaluate(best_model, test_dataloader, criterion, to_device)
        loss_test = loss_test.item()
        print('test loss {:8.5f} | test auroc {:8.3f} | test auprc {:8.3f}'.format(loss_test, auroc_test, auprc_test))
        
        res = {"model_name":model_name,
            "val_epoch_loss":val_epoch_loss, "train_epoch_loss":train_epoch_loss, "min_val_loss":min_val_loss,
                "loss_test":loss_test, "auc_test":auroc_test, "auprc_test": auprc_test, 
                "test_probs":prob_list.tolist(), "test_y":y_list.tolist(), 
                "test_subject_ids":subject_id_list}

        if prob_list.shape[1] > 2: #multiclass
            auroc_test, auroc_classes, auprc_test, auprc_classes, num_classes  = multiclass_auroc_auprc(y_list, prob_list)
            res.update({"num_classes": [int(c) for c in num_classes], "auroc_classes": auroc_classes, "auprc_classes": auprc_classes})
    else:
        loss_test, pred_list, y_list, subject_id_list = evaluate(best_model, test_dataloader, criterion, to_device, classification)
        loss_test = loss_test.item()
        res = {"model_name":model_name,
            "val_epoch_loss":val_epoch_loss, "train_epoch_loss":train_epoch_loss, "min_val_loss":min_val_loss,
                "loss_test":loss_test, 
                "test_preds":pred_list.tolist(), "test_y":y_list.tolist(), 
                "test_subject_ids":subject_id_list}
    
    del train_dataloader, valid_dataloader, test_dataloader
    gc.collect()
    torch.cuda.empty_cache()

    return best_model, optimizer, res

def save_model(best_model, optimizer, res, save_model_dir, fold_i, save_res = False):
    if os.path.exists(save_model_dir) == False:
        os.mkdir(save_model_dir)
    
    torch.save(best_model,'%s/model_%d.pth'%(save_model_dir, fold_i))
    
    if save_res:
        with open("%s/log_%d.json"%(save_model_dir,fold_i), "w") as f:
            json.dump(res, f)


def run_kfolds(train_param, model, dataset, kfolds, save = True, 
               collate_fun = None, log_folder = log_folder, classification = True):
    log_dir = log_folder+"/"+train_param["MODEL_NAME"]
    
    if save:
        if os.path.exists(log_dir) == False:
            os.mkdir(log_dir)
        save_model_dir = "%s/%s"%(log_dir,datetime.now().strftime("%y%m%d%H%M%S"))
    
    log = {"param":train_param}
    loss_weight = torch.tensor(train_param["LOSS_WEIGHT"]).to(device) if "LOSS_WEIGHT" in train_param else None
    to_device = True if collate_fun is None else False
    seed = train_param["RANDOM_SEED"] if "RANDOM_SEED" in train_param else 0
    all_preds = []
    all_y = []

    train_model = None
    best_model = None
    optimizer = None
    res = None
    
    try:
        for fold_i, (train_idx, val_idx, test_idx) in enumerate(kfolds):
            set_seed(seed)
            if type(model) == list:
                train_model = copy.deepcopy(model[fold_i])
            else:
                train_model = copy.deepcopy(model)
            best_model, optimizer, res = run(
                train_model, train_param["MODEL_NAME"], 
                dataset, train_param["BATCH_SIZE"], 
                train_idx, val_idx, test_idx,
                train_param["LR"], train_param["MAX_EPOCHS"],
                optimizer_type = train_param["OPTIMIZER"],  
                early_stop_thr=0.0001, early_cnt_thr=10,
                loss_weight = loss_weight,
                to_device = to_device,
                collate_fun = collate_fun,   
                random_seed = seed, 
                classification = classification                      
            )
            if classification:
                log["fold%d"%(fold_i)] = {k:float(res[k]) for k in ['min_val_loss', 'loss_test', 'auc_test', 'auprc_test']}
            else:
                log["fold%d"%(fold_i)] = {k:float(res[k]) for k in ['min_val_loss', 'loss_test']}
            if classification:
                all_preds.extend(res["test_probs"])
            else:
                all_preds.extend(res["test_preds"])
            
            all_y.extend(res["test_y"])
            if save:
                save_model(best_model, optimizer, res, save_model_dir, fold_i, True)

        if classification:
            if len(all_preds[0]) > 2: #multiclass
                auroc_test, auroc_classes, auprc_test, auprc_classes, num_classes  = multiclass_auroc_auprc(all_y, np.array(all_preds))
                log["auroc_classes"] = [float(v) for v in auroc_classes]
                log["auprc_classes"] = [float(v) for v in auprc_classes]
                log["num_classes"] = [int(c) for c in num_classes]
            else:
                auroc_test = roc_auc_score(all_y, np.array(all_preds)[:,1])
                precision, recall, thresholds = precision_recall_curve(all_y, np.array(all_preds)[:,1])
                auprc_test = auc(recall, precision)
            log["auroc_test"] = auroc_test
            log["auprc_test"] = auprc_test
        else:
            mse = np.mean((np.array(all_preds) - np.array(all_y))**2)
            log["mse_test"] = mse.astype(float)
            log["mae_test"] = np.mean(np.abs(np.array(all_preds) - np.array(all_y))).astype(float)

        if save:
            with open("%s/log.json"%(save_model_dir), "w") as f:
                json.dump(log, f)
            
            print(save_model_dir)
    
    except Exception as e:
        print("training error")
        traceback.print_exc()
        if save:
            with open("%s/log.json"%(save_model_dir), "w") as f:
                json.dump(log, f)
            
            with open("%s/error_log.txt"%(save_model_dir), "w") as f:
                f.write("Exception occurred during training:\n")
                traceback.print_exc(file=f)
        del model, train_model, best_model, optimizer, res, all_y, all_preds
        gc.collect()
        torch.cuda.empty_cache()
        return None

    del model, train_model, best_model, optimizer, res, all_y, all_preds
    gc.collect()
    torch.cuda.empty_cache()

    return log