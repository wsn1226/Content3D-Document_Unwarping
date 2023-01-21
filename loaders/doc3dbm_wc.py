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

class doc3dbm_wc(data.Dataset):
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
        #alb_path = pjoin('/home/sinan/DewarpNet-master', 'alb' , im_name + '.png')
        #d_path = pjoin('/home/sinan/DewarpNet-master', 'wc' , im_name + '.exr')
        wc_path=pjoin(self.root, 'wc_np', im_name + '.npy')
        #print(d_path)
        #img_foldr,fname=im_name.split('/')
        #recon_foldr='chess48'
        bm_path = pjoin('/home/sinan/DewarpNet-master', 'bm_np' , im_name + '.npy')
        #print(alb_path)
        #print(bm_path)
        #recon_path = pjoin(self.root,'recon' , img_foldr, fname[:-4]+recon_foldr+'0001.png')

        #timetostartIO=time.time()

        #alb = m.imread(alb_path,mode='RGB')
        #alb=cv2.imread(alb_path)[:,:,::-1]/255.0
        #wc=cv2.imread(d_path,cv2.IMREAD_ANYDEPTH)
        #timefinishalb=time.time()
        #wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        wc = np.load(wc_path)
        #print("??")
        wc = np.array(wc, dtype=np.float)
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
            wc, lbl = self.transform(wc,bm)
        #timetofinishtrans=time.time()

        #print("IO time:",timefinishedIO-timetostartIO)         
        #print("Preprocessing time:", timetofinishtrans-timetostarttrans) 
        return wc, lbl 

    def tight_crop(self,wc):
        # different tight crop
        #msk=(wc<10000).astype(np.uint8)
        msk=1-((wc[:,:,0]==0)&(wc[:,:,1]==0)&(wc[:,:,2]==0)).astype(np.uint8)
        size=msk.shape
        #wc=wc*msk
        [y, x] = (msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497   # calculate from all the wcs
        wc[:,:,0]= (wc[:,:,0]-zmn)/(zmx-zmn)
        wc[:,:,1]= (wc[:,:,1]-ymn)/(ymx-ymn)
        wc[:,:,2]= (wc[:,:,2]-xmn)/(xmx-xmn)
        wc=cv2.bitwise_and(wc,wc,mask=msk)

        #s = min(minx, miny, size[0]-(maxy + 1),size[1]-(maxx + 1),20)
        #im = im[miny: maxy, minx: maxx, :]
        wc = wc[miny: maxy, minx: maxx,:]
        
        s = 20
        #im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
        wc = np.pad(wc, ((s, s), (s, s), (0, 0)), 'constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        #im = im[cy1 : -cy2, cx1 : -cx2, :]
        wc= wc [cy1 : -cy2, cx1 : -cx2,:]
        #t=miny
        #b=size[0]-maxy
        #l=minx
        #r=size[1]-maxx
        #print(miny-s,minx-s)

        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2
        return wc,t,b,l,r


    def transform(self, wc, bm):
        #timestarttransform=time.time()
        wc,t,b,l,r=self.tight_crop(wc)               #t,b,l,r = is pixels cropped on top, bottom, left, right

        wc = cv2.resize(wc, self.img_size, interpolation=cv2.INTER_NEAREST)
        #wc = np.expand_dims(wc,axis=0) # HW -> CHW

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/np.array([448.0-l-r, 448.0-t-b])
        #bm*=128
        bm=(bm-0.5)*2

        bm0=cv2.resize(bm[:,:,0],(self.img_size[0],self.img_size[1]))
        bm1=cv2.resize(bm[:,:,1],(self.img_size[0],self.img_size[1]))
        lbl=np.stack([bm0,bm1],axis=-1)
        #timefinishtransform=time.time()
        #print("transform: ",timefinishtransform-timestarttransform)


        #alb = torch.from_numpy(alb).to(torch.float32).permute(2,0,1)
        wc=torch.from_numpy(wc).to(torch.float32).permute(2,0,1)
        #ensuremsk=(wc>=0).to(torch.float).repeat(3,1,1)
        #print(alb.shape,wc.shape)
        #rgbd=torch.cat([alb,wc],dim=0)
        lbl = torch.from_numpy(lbl).to(torch.float32)
        #timefinishtorch=time.time()
        #print("np to torch: ",timefinishtorch-timefinishtransform)
        return wc, lbl



 
# #Leave code for debugging purposes
if __name__ == '__main__':
    bs = 12
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3dbm_wc(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True, img_size=(128,128))
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

        #print(labels[0])
        #print(unwarpedimg.shape)
        #for i in range(10):
            #print(labels[0][i,0,:])
        for i in range(0,labels.shape[0]):
            cv2.imwrite('testwc'+str(i)+'.png',imgs[i,:3,:,:].permute(1,2,0).numpy()*255)
            #cv2.imwrite('testimg'+str(i)+'.png',imgs[i,:3,:,:].permute(1,2,0).numpy()*255)
            cv2.imwrite('testunwarp'+str(i)+'.png',unwarpedimg[i].permute(1,2,0).numpy()*255)
#print(prof.key_averages().table(sort_by="self_cpu_time_total"))
# 
# CUDA_VISIBLE_DEVICES=2 python doc3dbm_mobilevitfcn_rgbd_loader_norecon.py --resume /home/sinan/DewarpNet-master/checkpoints-bm5/mobilevit_sandfcnRGBD_9_0.0004073188203619793_0.00042028217193864514_bm_5_best_model.pkl
