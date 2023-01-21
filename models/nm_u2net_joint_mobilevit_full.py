from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2

class nm_u2net_joint_mobilevit_full(nn.Module):
    def __init__(self, boundary_model, nm_model, bm_model):
        super().__init__()
        self.boundary_model=boundary_model
        self.nm_model=nm_model
        self.bm_model=bm_model

    def forward(self, images, bm_img_size):
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
        pred_nms=self.nm_model(extracted_img)
        d1=pred_nms[0]
        d2=pred_nms[1]
        d3=pred_nms[2]
        d4=pred_nms[3]
        d5=pred_nms[4]
        d6=pred_nms[5]
        d7=pred_nms[6]
        #print(masked_nm.shape)
        #cv2.imwrite("testnm.png",masked_nm[0,:3,:,:].permute(1,2,0).detach().cpu().numpy()*255)
        d1*=mskorg
        d1pow = torch.pow(d1,2)
        #norm=torch.sqrt(torch.sum(torch.pow(d1,2),1)+1e-8)
        norm=torch.sqrt(torch.sum(d1pow,1)+1e-8)
        #print(torch.max(norm))
        msk=norm.nonzero()
        norm=norm.unsqueeze(1)
        input_nm = d1.clone()
        input_nm[msk[:,0],:,msk[:,1],msk[:,2]] = d1[msk[:,0],:,msk[:,1],msk[:,2]]/norm[msk[:,0],:,msk[:,1],msk[:,2]]
        #cv2.imwrite("testnm.png",d1[0,:3,:,:].permute(1,2,0).detach().cpu().numpy()*255)
        bm_input=F.interpolate(input_nm,bm_img_size,mode='bilinear',align_corners=True)
        pred_bm=self.bm_model(bm_input)

        return b1,b2,b3,b4,b5,b6,b7, d1,d2,d3,d4,d5,d6,d7,pred_bm