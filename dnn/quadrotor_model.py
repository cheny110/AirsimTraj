
from ast import mod
from numpy import dtype
import torch
import torch.nn as nn
print(f"Cuda available ?{torch.cuda.is_available()}")


class QuadrotorModel(nn.Module):
    def __init__(self,in_dim=16,out_dim=12,hidden_dim=52,hidden_layers=5,activation=nn.ReLU) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim =out_dim
        self.hidden_layers =hidden_layers
        self.hidden_dim = hidden_dim
        self.activation = activation()
        self.layers = nn.ModuleList()
        self._init_weight()
        
        first_layer = nn.Linear(in_dim,hidden_dim)
        self.layers.append(first_layer)
        for i in range(self.hidden_layers):
            hidden_layer = nn.Linear(hidden_dim,hidden_dim)
            self.layers.append(hidden_layer)
        out_layer =nn.Linear(hidden_dim,out_dim)
        self.layers.append(out_layer)
        
    def forward(self,x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x=self.layers[-1](x)
        return x
    
    def _init_weight(self):
        for layer in self.layers:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)   
                
if __name__ == "__main__":
    torch.cuda.set_device(0 if torch.cuda.is_available() else 'cpu')
    model =QuadrotorModel()
    print(model)
    total_neurons = model.in_dim+(model.hidden_layers*model.hidden_dim)+model.out_dim
    x = torch.Tensor([1.2,1.2,1.2,0,0,0,0,0,0,0,0,0,0,0,0,10.2]) #x,y,z,u,v,w,ax,ay,az, angular_velocityx,y,z, phi,theta psi, t
    res  = model(x)
    print(res)
    print(f"total_neurons:{total_neurons}")