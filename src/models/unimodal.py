import torch.nn as nn
import torch
import inspect
import glob
import importlib
import os
from config import src_folder

# scanning blocks
MODEL_DICT = {}
blocks_folder = os.path.join(src_folder, "blocks")
for module_path in glob.glob(os.path.join(blocks_folder, "*.py")):
    module_name = os.path.basename(module_path)[:-3]  # 去除 .py 後綴
    #import blocks.<module_name>
    module = importlib.import_module(f"blocks.{module_name}")
    # add all classes in the module to MODEL_DICT
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type):
            MODEL_DICT[attr_name] = attr

class UniModal(nn.Module):
    def __init__(self, encoder, decoder):
        super(UniModal, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        #get encoder out
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class MultiScale(nn.Module):
    def __init__(self, model_list, size_list):
        super(MultiScale, self).__init__()
        self.model_list = model_list
        self.size_list = size_list
    def forward(self, x):
        #B is batch size, T is the number of segments(time windows), S is the number of features
        B = x.size(0)
        T = x.size(1)
        x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3))
        multi_out = []
        onset = 0
        for model, size in zip(self.model_list, self.size_list):
            sub_x = x[:,:, onset:onset+size]
            multi_out.append(model(sub_x))
            onset += size
        out = torch.cat(multi_out, dim=1)
        out = out.view(B, T, -1)
        #transpose to (B, E, T)
        out = out.permute(0, 2, 1)
        return out
MODEL_DICT["MultiScale"] = MultiScale

class TemporalPooling(nn.Module):
    def __init__(self, embeds_model, win_size, device):
        super(TemporalPooling, self).__init__()
        self.embeds_model = embeds_model
        self.win_size = win_size
        self.device = device
    
    def forward(self, x):
        input = x[0]
        batch_num = x[1]
        batch_i = x[2]
        win_i = x[3]

        embeds = self.embeds_model(input)
        embeds_num = embeds.shape[-1]

        output = torch.zeros(batch_num, self.win_size, embeds_num, device=self.device)
        count = torch.zeros(batch_num, self.win_size, embeds_num, device=self.device)

        for i in range(len(batch_i)):
            b, w = batch_i[i], win_i[i]
            output[b, w] += embeds[i]
            count[b, w] += 1

        mask = count > 0
        output[mask] /= count[mask]
        
        #(B,T,E) -> (B,E,T)
        output = output.permute(0, 2, 1)
        return output
MODEL_DICT["TemporalPooling"] = TemporalPooling
    
def validate_args(func, args_dict):
    sig = inspect.signature(func)
    func_args = sig.parameters.keys()
    return {k:v for k,v in args_dict.items() if k in func_args}

def create_unimodal_model(encoder_model_list, encoder_param_list, decoder_model, decoder_param, device="cuda"):
    _encoder_model_list = []
    for model_name, model_param in zip(encoder_model_list, encoder_param_list):
        _encoder_model = MODEL_DICT[model_name]
        valid_model_param = validate_args(_encoder_model, model_param)
        _encoder = _encoder_model(**valid_model_param)
        _encoder_model_list.append(_encoder)
    encoder = nn.Sequential(*_encoder_model_list).to(device)
    
    _decoder_model = MODEL_DICT[decoder_model]
    decoder = _decoder_model(**decoder_param).to(device)
    
    model = UniModal(encoder, decoder).to(device)
    return model