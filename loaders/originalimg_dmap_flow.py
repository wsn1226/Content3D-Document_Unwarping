import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import cv2
import random
import time
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils import data

t1=torch.arange(-1,1,2/256)
grid_x, grid_y = torch.meshgrid(t1, t1)
originalgrid=torch.cat([grid_y.unsqueeze(-1),grid_x.unsqueeze(-1)],dim=-1).to(torch.float32)

npgridx=grid_x.numpy()
npgridy=grid_y.numpy()

from .augmentationsk import data_aug_full_joint,tight_crop_depth_joint_masked_depth

def color_jitter(im, brightness=0, contrast=0, saturation=0, hue=0):
    f = random.uniform(1 - contrast, 1 + contrast)
    im = np.clip(im * f, 0., 1.)
    f = random.uniform(-brightness, brightness)
    im = np.clip(im + f, 0., 1.).astype(np.float32)

    # hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    # f = random.uniform(-hue, hue)
    # hsv[0] = np.clip(hsv[0] + f * 360, 0., 360.)
    # f = random.uniform(-saturation, saturation)
    # hsv[2] = np.clip(hsv[2] + f, 0., 1.)
    # im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # im = np.clip(im, 0., 1.)
    return im

class originalimg_dmap_flow(data.Dataset):
    """
    Loader for world coordinate regression and RGB images
    """
    def __init__(self, root, split='train', is_transform=False,
                 img_size=512, augmentations=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 1   
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        for split in ['train', 'val']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        if self.augmentations:
            self.txpths=[]
            with open(os.path.join(self.root[:-7],'augtexnames.txt'),'r') as f:
                for line in f:
                    txpth=line.strip()
                    self.txpths.append(txpth)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                # 1/824_8-cp_Page_0503-7Nw0001
        #im_path = pjoin(self.root, 'img',  im_name + '.png')  
        #alb_path = pjoin('/home/sinan/DewarpNet-master', 'alb' , im_name + '.png')
        im_path = pjoin(self.root, 'img_np',  im_name + '.npy')  

        #labels 
        d_path = pjoin('/home/sinan/DewarpNet-master', 'dmap' , im_name + '.exr')
        bm_path = pjoin('/home/sinan/DewarpNet-master', 'bm_np' , im_name + '.npy')

        # Load data
        im=np.load(im_path) #RGB
        dmap=cv2.imread(d_path,cv2.IMREAD_ANYDEPTH)
        bm=np.load(bm_path)
        if 'val' in self.split:
            im, dmap,t,b,l,r=tight_crop_depth_joint_masked_depth(im/255.0,dmap)

        if self.augmentations:          #this is for training, default false for validation\
            tex_id=random.randint(0,len(self.txpths)-1)
            txpth=self.txpths[tex_id] 
            middle_path='doc3d/dtd/images_np/'
            #print(os.path.join(self.root[:-7],middle_path,txpth))
            bg=np.load(os.path.join(self.root[:-7],middle_path,txpth))
            #bg=cv2.resize(tex,self.img_size,interpolation=cv2.INTER_NEAREST)
            im,dmap,t,b,l,r=data_aug_full_joint(im,dmap,bg)

        if self.is_transform:
            im, msk, pmap, bm = self.transform(im, dmap, bm,t,b,l,r)
        #timefinishtransform=time.time()
        #print("transfome time: ",timefinishtransform-timestarttransform)
        return im, msk, pmap, bm

    def transform(self, img, dmap, bm,t,b,l,r):
        #img, dmap,t,b,l,r=self.tight_crop_boundary_joint_masked_depth(img,dmap)
        img = m.imresize(img, self.img_size) # uint8 with RGB mode
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1) # NHWC -> NCHW

        #normalize depth map
        dmap = cv2.resize(dmap, self.img_size, interpolation=cv2.INTER_NEAREST)
        dmap = np.expand_dims(dmap,axis=0)-1  # HW -> CHW

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/np.array([449.0-l-r, 449.0-t-b])
        bm=(bm-0.5)*2

        pmap0=bm[:,:,0]-npgridy
        pmap1=bm[:,:,1]-npgridx
        sign=np.sign(pmap0)
        pmap=np.sqrt(pmap0**2+pmap1**2)
        maxp=np.max(pmap)
        pmap=sign*(pmap/maxp)

        #bm0=cv2.resize(bm[:,:,0],(128,128))
        #bm1=cv2.resize(bm[:,:,1],(128,128))
        #bm=np.stack([bm0,bm1],axis=-1)

        msk=(dmap>=0).astype(np.uint8)
        # to torch
        img = torch.from_numpy(img).to(torch.float)
        msk = torch.from_numpy(msk).to(torch.float)
        pmap = torch.from_numpy(pmap).to(torch.float).unsqueeze(0)
        bm = torch.from_numpy(bm).to(torch.float)-originalgrid #HWC->CHW
        #print(torch.max(pmap),torch.min(pmap))
        #lbl=torch.cat([boundary,dmap,bm],dim=0)

        return img, msk, pmap, bm



if __name__ == '__main__':
    bs = 12
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = originalimg_dmap_flow(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True,augmentations=True, img_size=(256,256))
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        img, msk, pmap, bm = data
        #print(imgs.shape,labels.shape)
        #imgs,labels=imgs.permute(0,2,3,1).numpy(),labels.permute(0,2,3,1).numpy()
        unwarpedimg=F.grid_sample(img,bm+originalgrid)
        unwarpedimg=unwarpedimg.permute(0,2,3,1).numpy()

        for i in range(img.shape[0]):
            #cv2.imwrite('testimgmul'+str(i)+'.png',unwarpedimg[i]*255)
            cv2.imwrite('testrealpmap'+str(i)+'.png',pmap[i].permute(1,2,0).numpy()*255)
            cv2.imwrite('testrealmsk'+str(i)+'.png',msk[i].permute(1,2,0).numpy()*255)
        
