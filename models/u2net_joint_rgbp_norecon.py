from turtle import forward
from numpy import extract
import torch
import torch.nn.functional as F
import torch.nn as nn

class u2net_joint_rgbp_norecon(nn.Module):
    def __init__(self, boundary_model, prob_model, flow_model):
        super().__init__()
        self.boundary_model=boundary_model
        self.prob_model=prob_model
        self.flow_model=flow_model
        #self.relu=nn.ReLU()

    def forward(self, images, flow_img_size):
        #N,C,H,W=images.shape
        pred_boundaries=self.boundary_model(images[:,[2,1,0],:,:])
        b1=pred_boundaries[0]
        b2=pred_boundaries[1]
        b3=pred_boundaries[2]
        b4=pred_boundaries[3]
        b5=pred_boundaries[4]
        b6=pred_boundaries[5]
        b7=pred_boundaries[6]
        mskorg=(b1>0.5).to(torch.float32)
        extracted_img=torch.mul(mskorg.repeat(1,3,1,1),images)
        #recon=torch.mul(mskorg.repeat(1,3,1,1),recon)
        pred_prob=self.prob_model(extracted_img)
        prob=F.interpolate(pred_prob,flow_img_size,mode='bilinear',align_corners=True)
        rgb=F.interpolate(extracted_img,flow_img_size,mode='bilinear',align_corners=True)
        #recon=F.interpolate(recon,flow_img_size,mode='bilinear',align_corners=True)
        rgbp=torch.cat([rgb,prob],dim=1)
        #recond=torch.cat([recon,prob],dim=1)
        
        pred_flow=self.flow_model(rgbp)
        #pred_flowfromrecon=self.flow_model(recond)

        return b1,b2,b3,b4,b5,b6,b7,pred_prob,pred_flow