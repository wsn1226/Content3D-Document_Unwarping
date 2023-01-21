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

from .augmentationsk import nm_data_aug_full_joint

class doc3djoint_nm_full(data.Dataset):
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

    def tight_crop_val(self,im, norm):
        # different tight crop
        #print(norm.shape)
        msk=(1-((norm[:,:,0]==0)&(norm[:,:,1]==0)&(norm[:,:,2]==0))).astype(np.uint8)
        size=msk.shape
        #norm=norm*msk
        [y, x] = (msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)

        s = min(minx, miny, size[0]-(maxy + 1),size[1]-(maxx + 1),20)
        #s=20
        im = im[miny-s : maxy + 1 +s, minx-s : maxx + 1+s, :]
        norm = norm[miny-s : maxy + 1+s, minx-s : maxx + 1+s, :]
        
        #s = 20
        #im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
        #norm = np.pad(norm, ((s, s), (s, s)), 'constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        im = im[cy1 : -cy2, cx1 : -cx2, :]
        norm= norm[cy1 : -cy2, cx1 : -cx2, :]

        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2
        return im,norm,t,b,l,r

    def color_jitter(self, im, brightness=0, contrast=0, saturation=0, hue=0):
        f = random.uniform(1 - contrast, 1 + contrast)
        im = np.clip(im * f, 0., 1.)
        f = random.uniform(-brightness, brightness)
        im = np.clip(im + f, 0., 1.).astype(np.float32)

        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        #print(hsv.shape)
        f = random.uniform(-hue, hue)
        hsv[:,:,0] = np.clip(hsv[:,:,0] + f * 360, 0., 360.)
        #= random.uniform(-saturation, saturation)
        #hsv[:,:,1] = np.clip(hsv[:,:,1] + f, 0., 1.)
        im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        im = np.clip(im, 0., 1.)
        return im

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                # 1/824_8-cp_Page_0503-7Nw0001
        #im_path = pjoin(self.root, 'img',  im_name + '.png')  
        #alb_path = pjoin('/home/sinan/DewarpNet-master', 'alb' , im_name + '.png')
        im_path = pjoin(self.root, 'img_np',  im_name + '.npy')  

        #labels
        nm_path = pjoin(self.root, 'norm' , im_name + '.exr')
        #nm_path = pjoin('/home/sinan/DewarpNet-master', 'norm' , im_name + '.exr')
        bm_path = pjoin('/home/sinan/DewarpNet-master', 'bm_np' , im_name + '.npy')

        # Load data
        im=np.load(im_path) #RGB
        norm=cv2.imread(nm_path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        bm=np.load(bm_path)
        if 'val' in self.split:
            im, norm,t,b,l,r=self.tight_crop_val(im/255.0,norm)

        chance=random.random()
        if self.augmentations:          #this is for training, default false for validation\
            if chance>0.2:
                tex_id=random.randint(0,len(self.txpths)-1)
                txpth=self.txpths[tex_id] 
                middle_path='/home/sinan/DewarpNet-master/dtd/images_np'
                bg=np.load(os.path.join(middle_path,txpth))
                #bg=cv2.resize(tex,self.img_size,interpolation=cv2.INTER_NEAREST)
                im,norm,t,b,l,r=nm_data_aug_full_joint(im,norm,bg)
            else:
                im,norm,t,b,l,r=self.tight_crop_val(im/255.0,norm)
                im=self.color_jitter(im,0.2,0.2,0.6,0.6)

        if self.is_transform:
            im, norm, bm = self.transform(im, norm, bm,t,b,l,r)
        #timefinishtransform=time.time()
        #print("transfome time: ",timefinishtransform-timestarttransform)
        return im, norm, bm

    def transform(self, img, norm, bm,t,b,l,r):
        #img, norm,t,b,l,r=self.tight_crop_boundary_joint_masked_depth(img,norm)
        img = m.imresize(img, self.img_size) # uint8 with RGB mode
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1) # NHWC -> NCHW

        #normalize depth map
        norm = cv2.resize(norm, self.img_size, interpolation=cv2.INTER_NEAREST)
        norm = norm.transpose(2, 0, 1)  # HW -> CHW

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t-1
        bm[:,:,0]=bm[:,:,0]-l-1
        bm=bm/np.array([448.0-l-r, 448.0-t-b])
        bm=(bm-0.5)*2

        bm0=cv2.resize(bm[:,:,0],(128,128))
        bm1=cv2.resize(bm[:,:,1],(128,128))
        bm=np.stack([bm0,bm1],axis=-1)

        # to torch
        img = torch.from_numpy(img).to(torch.float)
        norm = torch.from_numpy(norm).to(torch.float)
        bm = torch.from_numpy(bm).to(torch.float) #HWC->CHW
        #lbl=torch.cat([boundary,norm,bm],dim=0)

        return img, norm, bm



if __name__ == '__main__':
    bs = 12
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3djoint_nm_full(root='/disk2/sinan/doc3d/', split='valmini', is_transform=True,augmentations=False, img_size=(128,128))
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        img, norm, bm = data
        #print(imgs.shape,labels.shape)
        #imgs,labels=imgs.permute(0,2,3,1).numpy(),labels.permute(0,2,3,1).numpy()
        unwarpedimg=F.grid_sample(img,bm)
        unwarpedimg=unwarpedimg.permute(0,2,3,1).numpy()

        for i in range(img.shape[0]):
            cv2.imwrite('testunwarp'+str(i)+'.png',unwarpedimg[i]*255)
            cv2.imwrite('testnormbi'+str(i)+'.png',norm[i].permute(1,2,0).numpy()*255)
            cv2.imwrite('testimg'+str(i)+'.png',img[i].permute(1,2,0).numpy()*255)
        
