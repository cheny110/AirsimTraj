

import torch.nn as nn
import numpy as np
import torch

class StateLoss(nn.Module):
    def __init__(self) -> None:
        super(StateLoss,self).__init__()
        
    def forward(self,predictions,labels):
        losses =0
        for prediction,label in zip(predictions,labels):
            predict_position = prediction[:3]
            label_position =label[:3]
            losses += torch.mean((predict_position-label_position)**2)*0.4
            predict_velocity = prediction[3:6]
            label_velocity = label[3:6] 
            losses += torch.mean((predict_velocity-label_velocity)**2)*0.4
            predict_orientation = prediction[6:9]
            label_orientation = label[6:9]
            losses += torch.mean((predict_orientation-label_orientation)**2)*0.1
            prediction_angluarity =prediction[9:12]
            label_angularity = label[9:12]
            losses += torch.mean((prediction_angluarity-label_angularity)**2)*0.1
        avg_loss=losses/len(predictions)
        return avg_loss
        
        