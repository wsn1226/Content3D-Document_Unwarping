# loader for backward mapping 
# loads nmedo to dewarp
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

class doc3dbm_mobilevitfcn_normal_loader_norecon(data.Dataset):
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
        nm_path = pjoin('/home/sinan/DewarpNet-master', 'norm' , im_name + '.exr')
        #img_foldr,fname=im_name.split('/')
        #recon_foldr='chess48'
        #bm_path = pjoin(self.root, 'bm' , im_name + '.mat')
        bm_path = pjoin(self.root, 'bm' , im_name + '.mat')
        #print(nm_path)
        #print(bm_path)
        #recon_path = pjoin(self.root,'recon' , img_foldr, fname[:-4]+recon_foldr+'0001.png')

        #timetostartIO=time.time()

        #nm = m.imread(nm_path,mode='RGB')

        nm = cv2.imread(nm_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        #nm=np.load(nm_path)
        #timefinishednm=time.time()
        #print("normal time: ",timefinishednm-timetostartIO)
        bm = h5.loadmat(bm_path)['bm']
        #bm=np.load(bm_path)
        #recon = m.imread(recon_path,mode='RGB')
        #recon = m.imread(recon_path)[:,:,::-1]/255

        #timefinishedbm=time.time()
        #print("loadbm time: ",timefinishedbm-timefinishednm)
        #nm = m.imread(nm_path,mode='RGB')
        #bm = h5.loadmat(bm_path)['bm']

        #timetostarttrans=time.time()       
        if self.is_transform:
            nm, lbl = self.transform(nm,bm)
        #timetofinishtrans=time.time()

        #print("IO time:",timefinishedIO-timetostartIO)         
        #print("Preprocessing time:", timetofinishtrans-timetostarttrans) 
        return nm, lbl
    


    def tight_crop(self, nm):
        #msk=((recon[:,:,0]!=0)&(recon[:,:,1]!=0)&(recon[:,:,2]!=0)).astype(np.uint8)
        msk=((nm[:,:,0]!=0)&(nm[:,:,1]!=0)&(nm[:,:,2]!=0)).astype(np.uint8)
        #print(msk)
        #msk=np.bitwise_not(msk)
        #print(msk)
        size=msk.shape
        [y,x]=(msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        nm = nm[miny : maxy + 1, minx : maxx + 1, :]
        #print("nmshape",nm.shape)
        #recon = recon[miny : maxy + 1, minx : maxx + 1, :]
        
        s = 20
        nm = np.pad(nm, ((s, s), (s, s), (0, 0)), mode='constant')
        #recon = np.pad(recon, ((s, s), (s, s), (0, 0)), 'constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        nm = nm[cy1 : -cy2, cx1 : -cx2, :]
        #recon = recon[cy1 : -cy2, cx1 : -cx2, :]
        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2

        return nm,t,b,l,r


    def transform(self, nm, bm):
        nm,t,b,l,r=self.tight_crop(nm)               #t,b,l,r = is pixels cropped on top, bottom, left, right
        #print(nm.shape,t,b,l,r)
        #cv2.imwrite('testrecon.png',recon)
        #cv2.imwrite('testimg.png',nm.cpu().numpy())

        
        nm = cv2.resize(nm, self.img_size)
        nm = nm.transpose(2, 0, 1) # NHWC -> NCHW

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/np.array([448.0-l-r, 448.0-t-b])
        bm=(bm-0.5)*2

        bm0=cv2.resize(bm[:,:,0],(self.img_size[0],self.img_size[1]))
        bm1=cv2.resize(bm[:,:,1],(self.img_size[0],self.img_size[1]))
        lbl=np.stack([bm0,bm1],axis=-1)

        nm = torch.from_numpy(nm).to(torch.float32)
        lbl = torch.from_numpy(lbl).to(torch.float32)
        return nm, lbl



 
# #Leave code for debugging purposes
if __name__ == '__main__':
    bs = 100
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3dbm_mobilevitfcn_normal_loader_norecon(root='/disk2/sinan/doc3d/', split='train', is_transform=True, img_size=(128,128))
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        print(imgs.shape,labels.shape)
        #print(tobeunwarped.shape)
        unwarpedimg=F.grid_sample(imgs[:,:,:,:],labels[:,:,:,:])
        print(labels[0])
        #print(unwarpedimg.shape)
        #for i in range(0,unwarpedimg.shape[0]):
            #cv2.imwrite('test'+str(i)+'.png',unwarpedimg[i].permute(1,2,0).cpu().numpy()*255)
#print(prof.key_averages().table(sort_by="self_cpu_time_total"))        
