

from audioop import bias
import os
import numpy as np
from rich.console import Console
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
from torch.utils.data.dataset import Dataset

class AirsimDataset(Dataset):
    def __init__(self,data_file,mode="train",bias=40,logger:Console=None) -> None:
        super(AirsimDataset,self).__init__()
        self.logger=logger
        self.bias =bias
        if not os.path.exists(data_file):
            self.logger.log(f"Can't load dataset from file {data_file},file not exists!!!",style="red")
        data=np.load(data_file,allow_pickle=True)
        split_len = int(data.shape[0]*0.7)
        if mode =="train":
            self.data = data[:split_len]
        elif mode == "test":
            self.data = data[split_len:]
    
    def __getitem__(self, index):
        data = self.data[index:self.bias+index]
        gt  = self.data[self.bias+index][:12]
        return data,gt    
    def __len__(self):
        return len(self.data)-bias
    
