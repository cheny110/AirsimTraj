
from scipy import optimize
from torch import mode
import torch
from quadrotor_model import QuadrotorModel
from toml import TomlDecoder
from tensorboardX.visdom_writer import VisdomWriter

def train(model:QuadrotorModel,train_loader,train_config,writer):
    model.train()
    train_loss =0
    optimize = torch.optim.Adam(model.parameters(),train_config["lr"])
    for i, input,target in enumerate(train_loader):
        

