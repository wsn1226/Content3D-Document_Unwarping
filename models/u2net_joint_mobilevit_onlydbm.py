from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn

class u2net_joint_mobilevit_onlydbm(nn.Module):
    def __init__(self, depth_model, bm_model):
        super().__init__()
        #self.boundary_model=boundary_model
        self.depth_model=depth_model
        self.bm_model=bm_model
        self.relu=nn.ReLU()

    def forward(self, mask, images, bm_img_size):
        #N,C,H,W=images.shape
        extracted_img=torch.mul(mask.repeat(1,3,1,1),images)
        pred_depths=self.depth_model(extracted_img)
        d1=self.relu(pred_depths[0])
        d2=self.relu(pred_depths[1])
        d3=self.relu(pred_depths[2])
        d4=self.relu(pred_depths[3])
        d5=self.relu(pred_depths[4])
        d6=self.relu(pred_depths[5])
        d7=self.relu(pred_depths[6])
        masked_depth=mask*d1
        rgbd=torch.cat([extracted_img,masked_depth],dim=1)
        bm_input=F.interpolate(rgbd,bm_img_size,mode='nearest')
        pred_bm=self.bm_model(bm_input)

        del masked_depth, rgbd, bm_input, extracted_img, pred_depths
        return d1,d2,d3,d4,d5,d6,d7,pred_bm