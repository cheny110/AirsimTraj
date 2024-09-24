

from math import inf
import toml
import torch
from quadrotor_model import QuadrotorModel
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from criteria import StateLoss
import os
from rich.console import Console
from dataloader import AirsimDataset
import time
logger=Console()
train_record ={
        "best_loss": inf,
        "best_model_name":""
    }

def validate(model:QuadrotorModel,test_loader,writer):
    global train_record
    model.eval()
    val_loss =0
    criterion = StateLoss()
    for i, input,gt in enumerate(test_loader):
        with torch.no_grad():
            input = Variable(input).cuda()
            gt= Variable(input).cuda()
            output =model(input)
            val_loss += criterion(output,gt)
    val_loss/=len(test_loader)
    writer.add_scalar("val_loss",val_loss)
    if train_record["best_loss"]>= val_loss:
        model_name = "model_best_{val_loss:.1f}".pth
        save_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),f"data/{model_name}")
        logger.log(f"Update best saved model, {model_name}",style="blue on white")
        train_record["best_loss"] = val_loss
        torch.save(model,model_name)

def train(model:QuadrotorModel,train_loader,test_loader,train_config,writer:SummaryWriter):
    global train_record
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),train_config["lr"],weight_decay=train_config["decay_rate"])
    num_epochs = train_config["epochs"]
    decay_start_epoch =train_config["lr_decay_start"]
    scheduler = StepLR(optimizer,step_size=train_config["num_epoch_lr_decay"])
    
    criterion = StateLoss()
    print_frequency = train_config["print_frequency"]
    validate_frequency =train_config["val_frequency"]
    i_tb=0 #tensorboard iteration
    for epoch in range(num_epochs):
        train_loss =0
        if epoch > decay_start_epoch:
            scheduler.step()
        for i, input,target in enumerate(train_loader):
            input = Variable(input).cuda()
            target =Variable(target).cuda()
            optimizer.zero_grad()
            output = model(input)
            loss =criterion(output,target)
            loss.backward()
            optimizer.step()
            train_loss +=loss.item()
        if epoch % print_frequency ==0:
            i_tb += 1
            logger.log(f"epoch:{epoch}, loss:{train_loss}")
            writer.add_scalar("train_loss",loss,i_tb)
        if epoch % validate_frequency ==0:
            validate(model,test_loader,train_record,writer)
        
            
if __name__ =="__main__":
    config_file = os.path.join(os.path.dirname(__file__),"config/config.toml")
    config =toml.load(config_file)
    model = QuadrotorModel()
    writer =SummaryWriter(log_dir=os.path.join(os.path.dirname(os.path.dirname(__file__),"exp")),
                        filename_suffix=time.strftime("%d-%M-%s",time.time()))
    train_loader = AirsimDataset(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/dnn_record_dataset.npy"),
                                 mode="train",
                                 logger=logger)
    test_loader = AirsimDataset(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/dnn_record_dataset.npy"),
                                mode="test",
                                logger=logger)
    train(model,train_loader,test_loader,config["train"],writer)
    