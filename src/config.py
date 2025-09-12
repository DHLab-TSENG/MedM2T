import torch
import sys
device = "cuda" if torch.cuda.is_available() else "cpu"

root_folder = "../../"
log_folder = root_folder+"logs/" 
data_folder = root_folder+"data/"
src_folder = root_folder+"MedM2T/src/"