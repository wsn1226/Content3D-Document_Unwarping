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

from .augmentationsk import data_aug_full_joint_withrecon_rgbp

t1=torch.arange(-1,1,2/256)
grid_x, grid_y = torch.meshgrid(t1, t1)
originalgrid=torch.cat([grid_y.unsqueeze(-1),grid_x.unsqueeze(-1)],dim=-1).to(torch.float32)
npgridx=grid_x.numpy()
npgridy=grid_y.numpy()

class doc3dbm_joint_rgbp_flow_loader_reconin(data.Dataset):
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

        for split in ['trainmini', 'valmini']:
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
        #d_path = pjoin('/home/sinan/DewarpNet-master', 'dmap' , im_name + '.exr')
        bm_path = pjoin('/home/sinan/DewarpNet-master', 'bm_np' , im_name + '.npy')
        recon_path = pjoin(self.root,'recon' , img_foldr, fname[:-4]+recon_foldr+'0001.png')

        # Load data
        im=np.load(im_path) #RGB
        #im=cv2.resize(im,(256,256))
        #dmap=cv2.imread(d_path,cv2.IMREAD_ANYDEPTH)
        #dmap=np.load(d_path)
        bm=np.load(bm_path)
        recon = cv2.imread(recon_path)
        #recon=np.load(recon_path)
        if 'val' in self.split:
            im, recon, t,b,l,r=self.tight_crop(im/255.0,recon/255.0)

        if self.augmentations:          #this is for training, default false for validation\
            tex_id=random.randint(0,len(self.txpths)-1)
            txpth=self.txpths[tex_id]
            middle_path='/home/sinan/DewarpNet-master/dtd/images_np'
            bg=np.load(os.path.join(middle_path,txpth))
            #bg=cv2.resize(tex,self.img_size,interpolation=cv2.INTER_NEAREST)
            im,recon,t,b,l,r=data_aug_full_joint_withrecon_rgbp(im,recon,bg)

        if self.is_transform:
            im, recon, pmap, bm = self.transform(im, recon, bm,t,b,l,r)
        #timefinishtransform=time.time()
        #print("transfome time: ",timefinishtransform-timestarttransform)
        return im, recon, pmap, bm

    def tight_crop(self, alb,recon):
        msk=((recon[:,:,0]==0)&(recon[:,:,1]==0)&(recon[:,:,2]==0)).astype(np.uint8)
        msk=1-msk
        #msk=((alb[:,:,0]!=0)&(alb[:,:,1]!=0)&(alb[:,:,2]!=0)).astype(np.uint8)
        #msk=((dmap<1000)).astype(np.uint8)
        #print(msk)
        #msk=np.bitwise_not(msk)
        #print(msk)
        size=msk.shape
        [y,x]=(msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        alb = alb[miny : maxy + 1, minx : maxx + 1, :]
        recon=recon[miny : maxy + 1, minx : maxx + 1, :]
        
        s = 20
        alb = np.pad(alb, ((s, s), (s, s), (0, 0)), mode='constant')
        recon = np.pad(recon, ((s, s), (s, s), (0, 0)), 'constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        alb = alb[cy1 : -cy2, cx1 : -cx2, :]
        recon= recon[cy1 : -cy2, cx1 : -cx2, :]
        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2
        return alb,recon,t,b,l,r

    def transform(self, img, recon, bm,t,b,l,r):
        #img, dmap,t,b,l,r=self.tight_crop_boundary_joint_masked_depth(img,dmap)
        img = m.imresize(img, self.img_size) # uint8 with RGB mode
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1) # NHWC -> NCHW

        recon = m.imresize(recon, self.img_size) # uint8 with RGB mode
        recon = recon.astype(float) / 255.0
        recon = recon.transpose(2, 0, 1) # NHWC -> NCHW

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/np.array([449.0-l-r, 449.0-t-b])
        bm=(bm-0.5)*2

        pmap0=bm[:,:,0]-npgridy
        pmap1=bm[:,:,1]-npgridx
        pmap=np.sqrt(pmap0**2+pmap1**2)
        maxp=np.max(pmap)
        pmap=pmap/maxp

        # to torch
        img = torch.from_numpy(img).to(torch.float)
        recon = torch.from_numpy(recon).to(torch.float)
        #dmap = torch.from_numpy(dmap).to(torch.float)
        pmap = torch.from_numpy(pmap).to(torch.float32).unsqueeze(0)
        bm = torch.from_numpy(bm).to(torch.float)-originalgrid #HWC->CHW
        #lbl=torch.cat([boundary,dmap,bm],dim=0)

        return img,recon,pmap,bm



if __name__ == '__main__':
    bs = 12
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3dbm_joint_rgbp_flow_loader_reconin(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True,augmentations=True, img_size=(256,256))
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        img,recon,pmap, bm = data
        #print(imgs.shape,labels.shape)
        #imgs,labels=imgs.permute(0,2,3,1).numpy(),labels.permute(0,2,3,1).numpy()
        unwarpedimg=F.grid_sample(img,bm+originalgrid)
        unwarpedimg=unwarpedimg.permute(0,2,3,1).numpy()

        for i in range(img.shape[0]):
            #cv2.imwrite('testunwarp'+str(i)+'.png',unwarpedimg[i]*255)
            cv2.imwrite('testpmap'+str(i)+'.png',pmap[i].permute(1,2,0).numpy()*255)
            #cv2.imwrite('testrealdmap'+str(i)+'.png',((dmap[i]!=0)).permute(1,2,0).numpy()*255)
            cv2.imwrite('testrecon'+str(i)+'.png',recon[i].permute(1,2,0).numpy()*255)
            #cv2.imwrite('testrealimg'+str(i)+'.png',img[i].permute(1,2,0).numpy()*255)
        
