from turtle import forward
from numpy import extract
import torch
import torch.nn.functional as F
import torch.nn as nn

class u2netdepth_joint_mobilevit_fullreconin(nn.Module):
    def __init__(self, boundary_model, depth_model, bm_model):
        super().__init__()
        self.boundary_model=boundary_model
        self.depth_model=depth_model
        self.bm_model=bm_model
        self.relu=nn.ReLU()

    def forward(self, images,recon, bm_img_size):
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
        recon=torch.mul(mskorg.repeat(1,3,1,1),recon)
        pred_depths=self.depth_model(extracted_img)
        d1=self.relu(pred_depths[0])
        d2=self.relu(pred_depths[1])
        d3=self.relu(pred_depths[2])
        d4=self.relu(pred_depths[3])
        d5=self.relu(pred_depths[4])
        d6=self.relu(pred_depths[5])
        d7=self.relu(pred_depths[6])
        masked_depth=mskorg*d1
        depth=F.interpolate(masked_depth,bm_img_size,mode='bilinear',align_corners=True)
        rgb=F.interpolate(extracted_img,bm_img_size,mode='bilinear',align_corners=True)
        recon=F.interpolate(recon,bm_img_size,mode='bilinear',align_corners=True)
        rgbd=torch.cat([rgb,depth],dim=1)
        recond=torch.cat([recon,depth],dim=1)
        
        pred_bm=self.bm_model(rgbd)
        pred_bmfromrecon=self.bm_model(recond)

        return b1,b2,b3,b4,b5,b6,b7, d1,d2,d3,d4,d5,d6,d7,pred_bm,pred_bmfromrecon