import torch.nn as nn
import torch
from blocks.attention import CrossAttn
from blocks.mlp import MLP, MLPDecoder
import inspect
import glob
import importlib
import os
import copy
from config import src_folder
from itertools import combinations

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


class BiModalAttn(nn.Module):
    def __init__(self,embed_size, num_blocks, num_heads, drop_prob, fusion_type, return_atten = False):
        super(BiModalAttn, self).__init__()
        
        self.blocks1 = nn.ModuleList([CrossAttn(embed_size, num_heads, drop_prob) for n in range(num_blocks)])
        self.blocks2 = nn.ModuleList([CrossAttn(embed_size, num_heads, drop_prob) for n in range(num_blocks)])
        
        self.fusion_type = fusion_type
        
        if fusion_type == "linear":
            self.linear = nn.Linear(embed_size*2, embed_size)
            
        self.return_atten = return_atten

    def forward(self, x1, x2):
        atten1 = x1
        for block in self.blocks1:
            atten1 = block(atten1, x2)
        
        atten2 = x2
        for block in self.blocks2:
            atten2 = block(atten2, x1)
            
        if self.fusion_type == "add":
            x = atten1+atten1
        elif self.fusion_type == "mul":
            x = torch.mul(atten1, atten1)
        elif self.fusion_type == "linear":
            x = torch.concat((atten1, atten2),dim = 1)
            x = self.linear(x)
        else:
            x = torch.concat((atten1, atten2),dim = 1)
        
        if self.return_atten:
            return x, atten1, atten2
        else:    
            return x
        
class MultiModal(nn.Module):
    def __init__(self, encoders, encoders_out_dim, emb_dim, 
                 bi_modal_list, bi_modal_interacts, shared_decoder, 
                 encoders_i = None, shared_layer = None):
        super(MultiModal, self).__init__()

        self.encoders = encoders
        
        #projection to emb_dim
        self.bi_modal_list = bi_modal_list
        self.proj_list = nn.ModuleList([nn.Linear(in_dim, emb_dim) for in_dim in encoders_out_dim])
        self.bi_modal_interacts = bi_modal_interacts
        self.shared_decoder = shared_decoder
        if encoders_i is None:
            self.encoders_i = list(range(len(encoders)))
        else:
            self.encoders_i = encoders_i
        self.shared_layer = shared_layer
        
    def forward(self, x):
        #get embeddings from modality-specific encoders
        embs = []
        for i in self.encoders_i:
            encoder = self.encoders[i]
            proj = self.proj_list[i]
            _emb = proj(encoder(x[i]))
            if self.shared_layer != None:
                _emb = self.shared_layer(_emb)
            embs.append(_emb)
        
        bi_interacts = []
        #get interaction between bi-modal
        for [i1,i2], bi_interact in zip(self.bi_modal_list, self.bi_modal_interacts):
            if self.encoders_i is not None:
                i1 = self.encoders_i.index(i1)
                i2 = self.encoders_i.index(i2)
            _emb1 = embs[i1]
            _emb2 = embs[i2]
            
            _interact = bi_interact(_emb1,_emb2)
            _atten1, _atten2 = None, None
            if type(_interact) is tuple:
                _interact, _atten1, _atten2 = _interact
        
            bi_interacts.append(_interact)
        
        #concatenate embeddings and interactions
        x = torch.cat(embs,dim=1)
        if len(bi_interacts) > 0:
            _bi_interacts = torch.cat(bi_interacts,dim=1)
            x = torch.cat((x, _bi_interacts),dim = 1)
            
        x = self.shared_decoder(x)
        return x
    
def validate_args(func, args_dict):
    sig = inspect.signature(func)
    func_args = sig.parameters.keys()
    return {k:v for k,v in args_dict.items() if k in func_args}

def create_multimodal_model(model_param, device, k_models = False, kfolds = 5):
    if  ("INTER_MODEL" not in model_param) or ("INTER_MODEL_PARAM" not in model_param):
        bi_modal_list = []
        bi_modal_interacts = []
    else:
        _inter_model = globals()[model_param["INTER_MODEL"]]
        bi_modal_list = list(combinations(model_param["ENCODERS_I"], 2))
        _bi_modal_interacts = _inter_model(**model_param["INTER_MODEL_PARAM"])  
        bi_modal_interacts = nn.ModuleList([copy.deepcopy(_bi_modal_interacts) for _ in bi_modal_list])

    _decoder = MODEL_DICT[model_param["DECODER_MODEL"]]
    shared_decoder = _decoder(**model_param["DECODER_PARAM"])

    shared_layer = None
    if "SHARED_LAYER_PARAM" in model_param:
        shared_layer = MLP(**model_param["SHARED_LAYER_PARAM"])

    _models = []
    if k_models:
        for i in range(kfolds):
            encoders_list = []
            encoders_out_dim = []
            for i, encoder_param in enumerate(model_param["ENCODERS_PARAM"]):
                pretrained_model = torch.load(encoder_param["model_path"]+"/model_%d.pth"%(i), weights_only=False)
                encoders_list.append(pretrained_model.encoder)
                encoders_out_dim.append(encoder_param["out_dim"])
            encoders = nn.ModuleList(encoders_list)

            model = MultiModal(
            encoders, encoders_out_dim, 
            model_param["EMBED_DIM"], bi_modal_list, 
            bi_modal_interacts, shared_decoder, 
            encoders_i=model_param["ENCODERS_I"],
            shared_layer=shared_layer).to(device)
            _models.append(model)
        return _models
    else:
        encoders_list = []
        encoders_out_dim = []
        for i, encoder_param in enumerate(model_param["ENCODERS_PARAM"]):
            pretrained_model = torch.load(encoder_param["model_path"], weights_only=False)
            encoders_list.append(pretrained_model.encoder)
            encoders_out_dim.append(encoder_param["out_dim"])
        encoders = nn.ModuleList(encoders_list)

        model = MultiModal(
        encoders, encoders_out_dim, 
        model_param["EMBED_DIM"], bi_modal_list, 
        bi_modal_interacts, shared_decoder, 
        encoders_i=model_param["ENCODERS_I"],
        shared_layer=shared_layer).to(device)
        return model