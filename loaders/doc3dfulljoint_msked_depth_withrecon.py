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

from .augmentationsk import data_aug_full_joint_withrecon

class doc3djoint_masked_depth_full_withrecon(data.Dataset):
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
        img_foldr,fname=im_name.split('/')
        recon_foldr='chess48'  

        #labels 
        d_path = pjoin('/home/sinan/DewarpNet-master', 'dmap' , im_name + '.exr')
        bm_path = pjoin('/home/sinan/DewarpNet-master', 'bm_np' , im_name + '.npy')
        recon_path = pjoin(self.root,'recon' , img_foldr, fname[:-4]+recon_foldr+'0001.png')

        # Load data
        im=np.load(im_path) #RGB
        #im=cv2.resize(im,(256,256))
        dmap=cv2.imread(d_path,cv2.IMREAD_ANYDEPTH)
        #dmap=np.load(d_path)
        bm=np.load(bm_path)
        recon = cv2.imread(recon_path)
        #recon=np.load(recon_path)
        if 'val' in self.split:
            im, recon,dmap, t,b,l,r=self.tight_crop(im/255.0,recon/255.0,dmap)

        if self.augmentations:          #this is for training, default false for validation\
            tex_id=random.randint(0,len(self.txpths)-1)
            txpth=self.txpths[tex_id]
            middle_path='/home/sinan/DewarpNet-master/dtd/images_np'
            bg=np.load(os.path.join(middle_path,txpth))
            #bg=cv2.resize(tex,self.img_size,interpolation=cv2.INTER_NEAREST)
            im,recon,dmap,t,b,l,r=data_aug_full_joint_withrecon(im,recon,dmap,bg)

        if self.is_transform:
            im, recon, dmap, bm = self.transform(im, recon, dmap, bm,t,b,l,r)
        #timefinishtransform=time.time()
        #print("transfome time: ",timefinishtransform-timestarttransform)
        return im, recon, dmap, bm

    def tight_crop(self, alb,recon, dmap):
        #msk=((recon[:,:,0]!=0)&(recon[:,:,1]!=0)&(recon[:,:,2]!=0)).astype(np.uint8)
        #msk=((alb[:,:,0]!=0)&(alb[:,:,1]!=0)&(alb[:,:,2]!=0)).astype(np.uint8)
        msk=((dmap<1000)).astype(np.uint8)
        dmap*=msk
        #print(msk)
        #msk=np.bitwise_not(msk)
        #print(msk)
        size=msk.shape
        [y,x]=(msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        maxd=np.max(dmap[msk.nonzero()])
        mind=np.min(dmap[msk.nonzero()])
        dmap[msk.nonzero()]=(dmap[msk.nonzero()]-mind)/(maxd-mind)+1
        alb = alb[miny : maxy + 1, minx : maxx + 1, :]
        recon=recon[miny : maxy + 1, minx : maxx + 1, :]
        #print("albshape",alb.shape)
        dmap = dmap[miny : maxy + 1, minx : maxx + 1]
        
        s = 20
        alb = np.pad(alb, ((s, s), (s, s), (0, 0)), mode='constant')
        recon = np.pad(recon, ((s, s), (s, s), (0, 0)), 'constant')
        dmap = np.pad(dmap, ((s, s), (s, s)), mode='constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        alb = alb[cy1 : -cy2, cx1 : -cx2, :]
        recon= recon[cy1 : -cy2, cx1 : -cx2, :]
        dmap = dmap[cy1 : -cy2, cx1 : -cx2]
        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2
        return alb,recon, dmap,t,b,l,r

    def transform(self, img, recon, dmap, bm,t,b,l,r):
        #img, dmap,t,b,l,r=self.tight_crop_boundary_joint_masked_depth(img,dmap)
        img = m.imresize(img, self.img_size) # uint8 with RGB mode
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1) # NHWC -> NCHW

        recon = m.imresize(recon, self.img_size) # uint8 with RGB mode
        recon = recon.astype(float) / 255.0
        recon = recon.transpose(2, 0, 1) # NHWC -> NCHW

        #normalize depth map
        dmap = cv2.resize(dmap, self.img_size)
        dmap = np.expand_dims(dmap,axis=0)-1  # HW -> CHW

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
        recon = torch.from_numpy(recon).to(torch.float)
        dmap = torch.from_numpy(dmap).to(torch.float)
        bm = torch.from_numpy(bm).to(torch.float) #HWC->CHW
        #lbl=torch.cat([boundary,dmap,bm],dim=0)

        return img,recon, dmap, bm



if __name__ == '__main__':
    bs = 12
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3djoint_masked_depth_full_withrecon(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True,augmentations=True, img_size=(192,192))
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        img,recon, dmap, bm = data
        #print(imgs.shape,labels.shape)
        #imgs,labels=imgs.permute(0,2,3,1).numpy(),labels.permute(0,2,3,1).numpy()
        unwarpedimg=F.grid_sample(img,bm)
        unwarpedimg=unwarpedimg.permute(0,2,3,1).numpy()

        for i in range(img.shape[0]):
            cv2.imwrite('testunwarp'+str(i)+'.png',unwarpedimg[i]*255)
            cv2.imwrite('testdmap'+str(i)+'.png',dmap[i].permute(1,2,0).numpy()*255)
            #cv2.imwrite('testrealdmap'+str(i)+'.png',((dmap[i]!=0)).permute(1,2,0).numpy()*255)
            cv2.imwrite('testrecon'+str(i)+'.png',recon[i].permute(1,2,0).numpy()*255)
            cv2.imwrite('testimg'+str(i)+'.png',img[i].permute(1,2,0).numpy()*255)
        
