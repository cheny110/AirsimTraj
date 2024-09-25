
from math import inf
import toml
import torch
from quadrotor_model import QuadrotorModel
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from criteria import StateLoss
import os
from rich.console import Console
from rich.progress import track
from dataset import AirsimDataset
from torch.utils.data import DataLoader
import time

logger=Console()

train_record ={
        "best_loss": inf,
        "best_model_name":""
    }

val_i_tb=0

def validate(model:QuadrotorModel,test_loader:DataLoader,writer,epoch):
    global train_record,val_i_tb
    model.eval()
    val_loss =0
    criterion = StateLoss()
    for i, data in enumerate(test_loader):
        with torch.no_grad():
            indata = torch.tensor(data[0]).view([test_loader.batch_size,-1]).cuda()
            gt= torch.tensor(data[1]).cuda()
            output =model(indata)
            val_loss += criterion(output,gt)
    val_loss/=len(test_loader)
    logger.log(f"validate once, epoch {epoch}, loss:{val_loss}",style="purple")
    val_i_tb+=1
    writer.add_scalar("val_loss",val_loss,val_i_tb)
    if train_record["best_loss"]>= val_loss:
        model_name = f"model_best_{val_loss:.1f}.pth"
        save_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),f"data/{model_name}")
        logger.log(f"Update best saved model, {model_name}",style="blue on white")
        train_record["best_loss"] = val_loss
        torch.save(model,os.path.join(os.path.dirname(os.path.dirname(__file__)),f"data/{model_name}"))

def train(model:QuadrotorModel,train_loader,test_loader,train_config,writer:SummaryWriter,logger:Console=logger):
    global train_record
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),train_config["lr"],weight_decay=train_config["decay_rate"])
    num_epochs = train_config["epochs"]
    decay_start_epoch =train_config["lr_decay_start"]
    scheduler = StepLR(optimizer,step_size=train_config["num_epoch_lr_decay"])
    
    criterion = StateLoss()
    print_frequency = train_config["print_frequency"]
    validate_frequency =train_config["val_frequency"]
    i_tb=0 #tensorboard iteration
    for epoch in track(range(1,num_epochs)):
        train_loss =0
        model.train()
        if epoch > decay_start_epoch:
            scheduler.step()
        for i, batch in enumerate(train_loader):
            inputs,labels=batch
            controls_in = torch.tensor(inputs,dtype=torch.float64,device="cuda").view([train_loader.batch_size,-1])
            labels =torch.tensor(labels,dtype=torch.float64,device="cuda")
            optimizer.zero_grad()
            output = model(controls_in)
            loss =criterion(output,labels)
            loss.backward()
            optimizer.step()
            train_loss +=loss.item()
            
        train_loss= len(train_loader)
        if epoch % print_frequency ==0:
            i_tb += 1
            logger.log(f"epoch:{epoch}, loss:{train_loss:.2f}")
            writer.add_scalar("train_loss",train_loss,i_tb)
        if epoch % validate_frequency ==0:
            validate(model,test_loader,writer,epoch)
        
            
if __name__ =="__main__":
    config_file = os.path.join(os.path.dirname(__file__),"config/config.toml")
    config =toml.load(config_file)
    model = QuadrotorModel()
    if config["train"]["resume"]: #resume training
        model_path = config["train"]["pretrain_model"]
        model_full_path =os.path.join(os.path.dirname(os.path.dirname(__file__)),model_path)
        if not os.path.exists(model_full_path):
            logger.log("Failed to resume training, saved model can't be found!!!",style="red on white")
        else:
            logger.log("Resume training...",style="green")
            pretrain_weight=torch.load(model_full_path)
            model.load_state_dict(pretrain_weight)
            model.eval()
            logger.log("Load pretrained weight successfully.",style="green")
    writer =SummaryWriter(os.path.join(os.path.dirname(os.path.dirname(__file__)),"exp",time.strftime("%d-%h-%M-%S",time.localtime())))
    train_dataset = AirsimDataset(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/dnn_record_dataset.npy"),
                                 mode="train",
                                 logger=logger)
    test_dataset= AirsimDataset(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/dnn_record_dataset.npy"),
                                mode="test",
                                logger=logger)
    batch_size = config["train"]["batch_size"]
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=config["train"]["dataload_workers"],drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=config["test"]["batch_size"],shuffle=True,num_workers=config["test"]["dataload_workers"],drop_last=False)
    train(model,train_loader,test_loader,config["train"],writer,logger)
    