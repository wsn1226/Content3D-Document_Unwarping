from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn

class u2netdepth_joint_mobilevit(nn.Module):
    def __init__(self, depth_model, bm_model):
        super().__init__()
        self.depth_model=depth_model
        self.bm_model=bm_model
        self.relu=nn.ReLU()

    def forward(self, images, mask, bm_img_size):
        N,C,H,W=images.shape
        pred_depths=self.depth_model(images)
        d1=self.relu(pred_depths[0])
        d2=self.relu(pred_depths[1])
        d3=self.relu(pred_depths[2])
        d4=self.relu(pred_depths[3])
        d5=self.relu(pred_depths[4])
        d6=self.relu(pred_depths[5])
        d7=self.relu(pred_depths[6])
        masked_depth=mask*d1
        rgbd=torch.cat([images,masked_depth],dim=1)
        bm_input=F.interpolate(rgbd,bm_img_size,mode='bilinear',align_corners=True)
        pred_bm=self.bm_model(bm_input)

        return d1,d2,d3,d4,d5,d6,d7,pred_bm