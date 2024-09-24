

import torch.nn as nn
import numpy as np

class StateLoss(nn.Module):
    def __init__(self) -> None:
        super(StateLoss,self).__init__()
        
    def forward(self,prediction,label):
        loss =0
        predict_position = np.array(prediction[:3])
        label_position =np.array(label[:3])
        loss += np.mean((predict_position-label_position)**2)*0.4
        predict_velocity = np.array(prediction[3:6])
        label_velocity =np.array(label[3:6])
        loss += np.mean((predict_velocity-label_velocity)**2)*0.4
        predict_orientation = np.array(label[6:9])
        label_orientation = np.array(label[6:9])
        loss += np.mean((predict_orientation-label_orientation)**2)*0.1
        prediction_angluarity =np.array(prediction[9:12])
        label_angularity = np.array(label[9:12])
        loss += np.mean((prediction_angluarity-label_angularity)**2)*0.1
        return loss
        
        