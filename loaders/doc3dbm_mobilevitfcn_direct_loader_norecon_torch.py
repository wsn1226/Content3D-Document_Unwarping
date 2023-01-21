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

class doc3dbm_mobilevitfcn_direct_loader_norecon(data.Dataset):
    """
    Data loader.
    """
    def __init__(self, root='/disk2/sinan/doc3d/', split='train', is_transform=False,
                 img_size=(256,256)):
        self.root=root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 2
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        for split in ['trainmini', 'valmini']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        #self.setup_annotations()


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                 #1/2Xec_Page_453X56X0001.png
        alb_path = pjoin(self.root, 'alb' , im_name + '.png')
        img_foldr,fname=im_name.split('/')
        recon_foldr='chess48'
        bm_path = pjoin(self.root, 'bm' , im_name + '.mat')
        #print(alb_path)
        #print(bm_path)
        recon_path = pjoin(self.root,'recon' , img_foldr, fname[:-4]+recon_foldr+'0001.png')

        timetostartIO=time.time()

        #alb = m.imread(alb_path,mode='RGB')
        alb=cv2.imread(alb_path)
        bm = h5.loadmat(bm_path)['bm']
        #recon = m.imread(recon_path,mode='RGB')
        recon = m.imread(recon_path)

        timefinishedIO=time.time()
        #alb = m.imread(alb_path,mode='RGB')
        #bm = h5.loadmat(bm_path)['bm']
        
        alb=torch.from_numpy(alb).to(torch.float32)
        bm=torch.from_numpy(bm).to(torch.float32)
        recon=torch.from_numpy(recon).to(torch.float32)/255.0

        timetostarttrans=time.time()       
        if self.is_transform:
            im, lbl = self.transform(alb,bm,recon)
        timetofinishtrans=time.time()

        print("IO time:",timefinishedIO-timetostartIO)         
        print("Preprocessing time:", timetofinishtrans-timetostarttrans) 
        return im, lbl
    


    def tight_crop(self, alb, recon):
        #msk=((recon[:,:,0]==0)&(recon[:,:,1]==0)&(recon[:,:,2]==0)).astype(np.uint8)
        msk=((recon[:,:,0]!=0)&(recon[:,:,1]!=0)&(recon[:,:,2]!=0)).to(torch.int8)
        #print(recon.shape)
        #msk=np.bitwise_not(msk)
        #msk=1-msk
        size=msk.shape
        print(msk.nonzero().shape)
        y,x=(msk).nonzero()[:,0],(msk).nonzero()[:,1]
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        alb = alb[miny : maxy + 1, minx : maxx + 1, :]
        #recon = recon[miny : maxy + 1, minx : maxx + 1, :]
        
        s = 20
        alb = F.pad(alb, (0,0,s,s,s,s), mode='constant')
        #recon = np.pad(recon, ((s, s), (s, s), (0, 0)), 'constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        alb = alb[cy1 : -cy2, cx1 : -cx2, :]
        #recon = recon[cy1 : -cy2, cx1 : -cx2, :]
        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2

        return alb,t,b,l,r


    def transform(self, alb, bm, recon):
        alb,t,b,l,r=self.tight_crop(alb, recon)               #t,b,l,r = is pixels cropped on top, bottom, left, right
        print(alb.shape,t,b,l,r)
        #cv2.imwrite('testrecon.png',recon)
        #cv2.imwrite('testimg.png',alb.cpu().numpy())

        alb=alb.unsqueeze(0).permute(0,3,1,2)
        alb = F.interpolate(alb, self.img_size,mode='bilinear')
        alb = alb / 255.0
        #print(alb.shape)
        #alb = alb.permute(2, 0, 1) # NHWC -> NCHW

        #bm = bm.to(torch.float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/torch.tensor([448.0-l-r, 448.0-t-b]).float()
        bm=(bm-0.5)*2


        bm=bm.unsqueeze(0).permute(0,3,1,2)
        print(bm.shape)
        bm0=F.interpolate(bm[:,:1,:,:],(self.img_size[0],self.img_size[1]),mode='bilinear')
        bm1=F.interpolate(bm[:,1:,:,:],(self.img_size[0],self.img_size[1]),mode='bilinear')
        lbl=torch.cat([bm0,bm1],dim=1)

        #img = torch.from_numpy(alb).to(torch.float32)
        #lbl = torch.from_numpy(lbl).to(torch.float32)
        return alb.squeeze(0), lbl.squeeze(0)



 
# #Leave code for debugging purposes
if __name__ == '__main__':
    bs = 16
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3dbm_mobilevitfcn_direct_loader_norecon(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True, img_size=(128,128))
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        print(imgs.shape,labels.shape)
        #print(tobeunwarped.shape)
        unwarpedimg=F.grid_sample(imgs,labels.permute(0,2,3,1))
        #print(unwarpedimg.shape)
        for i in range(0,unwarpedimg.shape[0]):
            #print(str(i))
            cv2.imwrite('test'+str(i)+'.png',unwarpedimg[i].permute(1,2,0).cpu().numpy()*255)
#print(prof.key_averages().table(sort_by="self_cpu_time_total"))        
