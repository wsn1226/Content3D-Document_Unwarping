from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2

class u2net_joint_mobilevit_onlydbm_rgbdiff(nn.Module):
    def __init__(self, depth_model, bm_model):
        super().__init__()
        #self.boundary_model=boundary_model
        self.depth_model=depth_model
        self.bm_model=bm_model
        self.relu=nn.ReLU()

    def forward(self, mask, images, bm_img_size):
        #N,C,H,W=images.shape
        extracted_img=torch.mul(mask.repeat(1,3,1,1),images)
        alb0 = extracted_img[:,:,1:,:]-extracted_img[:,:,:-1,:]
        alb1 = extracted_img[:,:,:,1:]-extracted_img[:,:,:,:-1]
        alb0 = F.pad(alb0, ((0,0,1,0,0,0,0,0)), mode='constant')
        alb1 = F.pad(alb1, ((1,0,0,0,0,0,0,0)), mode='constant')
        alb0 = torch.abs(F.interpolate(alb0, bm_img_size, mode='bilinear', align_corners=False))
        alb1 = torch.abs(F.interpolate(alb1, bm_img_size, mode='bilinear', align_corners=False))
        alb = (alb0+alb1).to(torch.float)
        maxalb=torch.max(alb)
        alb/=maxalb
        #cv2.imwrite('testalb.png',alb[0].permute(1,2,0).cpu().numpy()*255)
        pred_depths=self.depth_model(extracted_img)
        d1=self.relu(pred_depths[0])
        d2=self.relu(pred_depths[1])
        d3=self.relu(pred_depths[2])
        d4=self.relu(pred_depths[3])
        d5=self.relu(pred_depths[4])
        d6=self.relu(pred_depths[5])
        d7=self.relu(pred_depths[6])
        masked_depth=mask*d1
        bm_inputdepth=F.interpolate(masked_depth,bm_img_size,mode='nearest')
        rgbd=torch.cat([alb,bm_inputdepth],dim=1)
        #bm_input=F.interpolate(rgbd,bm_img_size,mode='nearest')
        pred_bm=self.bm_model(rgbd)

        del masked_depth, rgbd, extracted_img, pred_depths, bm_inputdepth, alb, alb0, alb1
        return d1,d2,d3,d4,d5,d6,d7,pred_bm