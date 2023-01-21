from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn

class stacked_u2net_depth(nn.Module):
    def __init__(self, d1_model, d2_model):
        super().__init__()
        self.d1_model=d1_model
        self.d2_model=d2_model
        self.relu=nn.ReLU()

    def forward(self, images):
        #N,C,H,W=images.shape
        pred_depth_1=self.d1_model(images)
        b1=self.relu(pred_depth_1[0])
        b2=self.relu(pred_depth_1[1])
        b3=self.relu(pred_depth_1[2])
        b4=self.relu(pred_depth_1[3])
        b5=self.relu(pred_depth_1[4])
        b6=self.relu(pred_depth_1[5])
        b7=self.relu(pred_depth_1[6])

        pred_depth_2=self.d2_model(b1)
        d1=self.relu(pred_depth_2[0])
        d2=self.relu(pred_depth_2[1])
        d3=self.relu(pred_depth_2[2])
        d4=self.relu(pred_depth_2[3])
        d5=self.relu(pred_depth_2[4])
        d6=self.relu(pred_depth_2[5])
        d7=self.relu(pred_depth_2[6])

        return b1,b2,b3,b4,b5,b6,b7,d1,d2,d3,d4,d5,d6,d7