# loader for backward mapping 
# loads albedo to dewarp
# uses crop as augmentation
import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import time
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt

import glob
import cv2
import hdf5storage as h5
import random
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils import data

class doc3dbm_mobilevitfcn_rgb_diff_loader_norecon(data.Dataset):
    """
    Data loader.
    """
    def __init__(self, root='/disk3/sinan/doc3d/', split='train', is_transform=False,
                 img_size=(256,256)):
        self.root=root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 2
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        for split in ['train', 'val']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        #self.setup_annotations()


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                 #1/2Xec_Page_453X56X0001.png
        alb_path = pjoin('/home/sinan/DewarpNet-master', 'alb' , im_name + '.png')
        d_path = pjoin('/home/sinan/DewarpNet-master', 'dmap' , im_name + '.exr')
        #print(d_path)
        #img_foldr,fname=im_name.split('/')
        #recon_foldr='chess48'
        bm_path = pjoin('/home/sinan/DewarpNet-master', 'bm_np' , im_name + '.npy')
        #print(alb_path)
        #print(bm_path)
        #recon_path = pjoin(self.root,'recon' , img_foldr, fname[:-4]+recon_foldr+'0001.png')

        #timetostartIO=time.time()

        #alb = m.imread(alb_path,mode='RGB')/255.0
        alb=cv2.imread(alb_path)[:,:,::-1]/255.0
        dmap=cv2.imread(d_path,cv2.IMREAD_ANYDEPTH)
        #timefinishalb=time.time()
        #print("alb: ",timefinishalb-timetostartIO)
        #bm = h5.loadmat(bm_path)['bm']
        bm=np.load(bm_path)
        #recon = m.imread(recon_path,mode='RGB')
        #recon = m.imread(recon_path)[:,:,::-1]/255

        #timefinishedIO=time.time()
        #print("bm: ",timefinishedIO-timefinishalb)
        #alb = m.imread(alb_path,mode='RGB')
        #bm = h5.loadmat(bm_path)['bm']

        #timetostarttrans=time.time()       
        if self.is_transform:
            rgbd, lbl = self.transform(alb,dmap,bm)
        #timetofinishtrans=time.time()

        #print("IO time:",timefinishedIO-timetostartIO)         
        #print("Preprocessing time:", timetofinishtrans-timetostarttrans) 
        return rgbd, lbl

    def color_jitter(self, im, brightness=0, contrast=0, saturation=0, hue=0):
        f = random.uniform(1 - contrast, 1 + contrast)
        im = np.clip(im * f, 0., 1.)
        f = random.uniform(-brightness, brightness)
        im = np.clip(im + f, 0., 1.).astype(np.float32)

        #hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        #f = random.uniform(-hue, hue)
        #hsv[:,:,0] = np.clip(hsv[:,:,0] + f * 360, 0., 360.)
        
        #f = random.uniform(-saturation, saturation)
        #hsv[:,:,1] = np.clip(hsv[:,:,1] + f, 0., 1.)
        #im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        im = np.clip(im, 0., 1.)
        return im    

    def tight_crop(self, alb, dmap):
        #msk=((recon[:,:,0]!=0)&(recon[:,:,1]!=0)&(recon[:,:,2]!=0)).astype(np.uint8)
        msk=((alb[:,:,0]!=0)&(alb[:,:,1]!=0)&(alb[:,:,2]!=0)).astype(np.uint8)
        #print(msk)
        #msk=np.bitwise_not(msk)
        #print(msk)
        size=msk.shape
        dmap=dmap*msk
        [y,x]=(msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        maxd=np.max(dmap[msk.nonzero()])
        mind=np.min(dmap[msk.nonzero()])
        dmap[msk.nonzero()]=(dmap[msk.nonzero()]-mind)/(maxd-mind)
        alb = alb[miny : maxy + 1, minx : maxx + 1, :]
        #print("albshape",alb.shape)
        dmap = dmap[miny : maxy + 1, minx : maxx + 1]
        
        s = 20
        alb = np.pad(alb, ((s, s), (s, s), (0, 0)), mode='constant')
        dmap = np.pad(dmap, ((s, s), (s, s)), mode='constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        alb = alb[cy1 : -cy2, cx1 : -cx2, :]
        dmap = dmap[cy1 : -cy2, cx1 : -cx2]
        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2
        #msk_alb=1-((alb[:,:,0]==0)&(alb[:,:,1]==0)&(alb[:,:,2]==0)).astype(np.uint8)
        #if 'train' in self.split:
            #alb=self.color_jitter(alb,0.2,0.2,0.6,0.6)
            #alb*=np.repeat(np.expand_dims(msk_alb,-1),3,axis=-1)
        return alb,dmap,t,b,l,r


    def transform(self, alb,dmap, bm):
        #timestarttransform=time.time()
        alb,dmap,t,b,l,r=self.tight_crop(alb,dmap)               #t,b,l,r = is pixels cropped on top, bottom, left, right
        #print(alb.shape,t,b,l,r)
        #cv2.imwrite('testrecon.png',recon)
        #cv2.imwrite('testimg.png',alb.cpu().numpy())
        #timefinishcrop=time.time()
        #print((dmap<0).astype(np.uint8).nonzero())
        #print("crop time: ",timefinishcrop-timestarttransform)

        alb0 = np.diff(alb,axis=0)
        alb1 = np.diff(alb,axis=1)
        alb0 = np.pad(alb0, ((1,0), (0,0), (0, 0)), mode='constant')
        alb1 = np.pad(alb1, ((0,0), (1,0), (0, 0)), mode='constant')
        #print(alb0.shape,alb1.shape)
        alb0 = np.abs(cv2.resize(alb0, self.img_size))
        alb1 = np.abs(cv2.resize(alb1, self.img_size))
        #alb0=torch.from_numpy(alb0).permute(2,0,1).unsqueeze(0)
        #alb1=torch.from_numpy(alb1).permute(2,0,1).unsqueeze(0)
        #alb0=F.interpolate(alb0, (128,128), mode='bilinear', align_corners=False).squeeze(0)
        #alb1=F.interpolate(alb1, (128,128), mode='bilinear', align_corners=False).squeeze(0)
        #alb0=np.abs(alb0.numpy())
        #alb1=np.abs(alb1.numpy())
        #alb0 = np.abs(m.imresize(alb0, self.img_size))/255.0
        #alb1 = np.abs(m.imresize(alb1, self.img_size))/255.0
        diff = (alb0+alb1).astype(float)
        maxdiff=np.max(diff)
        diff/=maxdiff
        #print(alb.shape)
        diff*=255
        diffgray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)/255.0

        alb = m.imresize(alb, self.img_size)
        alb = alb.astype(float)/255.0
        alb = alb.transpose(2, 0, 1) # HWC -> CHW
        #alb = alb.transpose(2, 0, 1) # HWC -> CHW

        #dmap = cv2.resize(dmap, self.img_size,interpolation=cv2.INTER_NEAREST)
        #dmap = np.expand_dims(dmap,axis=0) # HW -> CHW

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t-1
        bm[:,:,0]=bm[:,:,0]-l-1
        bm=bm/np.array([448.0-l-r, 448.0-t-b])
        bm=(bm-0.5)*2

        bm0=cv2.resize(bm[:,:,0],(self.img_size[0],self.img_size[1]))
        bm1=cv2.resize(bm[:,:,1],(self.img_size[0],self.img_size[1]))
        lbl=np.stack([bm0,bm1],axis=-1)
        #timefinishtransform=time.time()
        #print("transform: ",timefinishtransform-timestarttransform)

        #alb = np.pad(alb, ((s, s), (s, s), (0, 0)), mode='constant')
        alb = torch.from_numpy(alb).to(torch.float32)
        diffgray=torch.from_numpy(diffgray).to(torch.float32).unsqueeze(0)
        rgbd=torch.cat([alb,diffgray],dim=0)
        lbl = torch.from_numpy(lbl).to(torch.float32)
        #timefinishtorch=time.time()
        #print("np to torch: ",timefinishtorch-timefinishtransform)
        return rgbd, lbl



 
# #Leave code for debugging purposes
if __name__ == '__main__':
    bs = 12
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3dbm_mobilevitfcn_rgb_diff_loader_norecon(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True, img_size=(256,256))
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data

        #timestartcuda=time.time()
        #imgs=imgs.cuda()
        #labels=labels.cuda()
        #timefinishcuda=time.time()
        #print("Cuda time: ",timefinishcuda-timestartcuda)
        #print(imgs.shape,labels.shape)
        #print((imgs[:,3:,:,:]<0).to(torch.int8).nonzero())
        unwarpedimg=F.grid_sample(imgs[:,:3,:,:],labels[:,:,:,:])
        #imgs=F.interpolate(imgs[:,:3,:,:], (128,128), mode='bilinear', align_corners=False)

        #print(labels[0])
        #print(unwarpedimg.shape)
        #for i in range(10):
            #print(labels[0][i,0,:])
        for i in range(0,labels.shape[0]):
            cv2.imwrite('testdmap'+str(i)+'.png',imgs[i,3:,:,:].permute(1,2,0).numpy()*255)
            #cv2.imwrite('testimgdiffmread'+str(i)+'.png',imgs[i,:3,:,:].permute(1,2,0).numpy()*255)
            #cv2.imwrite('testmsk'+str(i)+'.png',((imgs[i,0,:,:]<0.1)&(imgs[i,1,:,:]<0.1)&(imgs[i,2,:,:]<0.1)).numpy()*255)
            #cv2.imwrite('testunwarp449'+str(i)+'.png',unwarpedimg[i].permute(1,2,0).numpy()*255)
#print(prof.key_averages().table(sort_by="self_cpu_time_total"))
# 
# CUDA_VISIBLE_DEVICES=2 python doc3dbm_mobilevitfcn_rgbd_loader_norecon.py --resume /home/sinan/DewarpNet-master/checkpoints-bm5/mobilevit_sandfcnRGBD_9_0.0004073188203619793_0.00042028217193864514_bm_5_best_model.pkl
