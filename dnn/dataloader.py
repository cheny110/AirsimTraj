

import os
import numpy as np
from rich.console import Console
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
from torch.utils.data.dataset import Dataset

class AirsimDataset(Dataset):
    def __init__(self,data_file,logger:Console=None) -> None:
        super(AirsimDataset,self).__init__()
        self.logger=logger
        if not os.path.exists(data_file):
            self.logger.log(f"Can't load dataset from file {data_file},file not exists!!!",style="red")
        self.data=np.load(data_file,allow_pickle=True)
    
    def __getitem__(self, index):
        data = self.data[index]
        return data[:]
    
    def __len__(self):
        return len(self.data)
    
