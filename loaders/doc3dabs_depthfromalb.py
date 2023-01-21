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

class doc3dabs_depthfromalb(data.Dataset):
    """
    Data loader.
    """
    def __init__(self, root='/disk2/sinan/doc3d/', split='train', is_transform=False,
                 img_size=(256,256)):
        self.root=root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 1
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
        lbl_path=pjoin('/home/sinan/DewarpNet-master', 'dmap', im_name + '.exr')
        #img_foldr,fname=im_name.split('/')
        #img_foldr,fname=im_name.split('/')
        #recon_foldr='chess48'
        #recon_foldr='chess48'
        #print(alb_path)
        #print(lbl_path)
        #recon_path = pjoin(self.root,'recon' , img_foldr, fname[:-4]+recon_foldr+'0001.png')

        #timetostartIO=time.time()

        #alb = m.imread(alb_path,mode='RGB')
        alb=cv2.imread(alb_path)[:,:,::-1]/255.0 #RGB
        lbl = cv2.imread(lbl_path, cv2.IMREAD_ANYDEPTH)
        #timefinishalb=time.time()
        #print("alb: ",timefinishalb-timetostartIO)

        #recon = m.imread(recon_path,mode='RGB')
        #recon = cv2.imread(recon_path)[:,:,::-1]

        #timefinishedIO=time.time()
        #print("lbl: ",timefinishedIO-timefinishalb)
        #alb = m.imread(alb_path,mode='RGB')
        #lbl = h5.loadmat(lbl_path)['lbl']

        #timetostarttrans=time.time()       
        if self.is_transform:
            im, lbl = self.transform(alb,lbl)
        #timetofinishtrans=time.time()

        #print("IO time:",timefinishedIO-timetostartIO)         
        #print("Preprocessing time:", timetofinishtrans-timetostarttrans) 
        return im, lbl
    


    def tight_crop_d(self, im, dm):
        # different tight crop
        msk=(dm<1000).astype(np.uint8)
        dm=dm*msk
        [y, x] = (msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        xmx, xmn= 4.8, 1.8# calculate from all the depths
        
        thismax = np.max(dm)
        thismin = np.min(dm)
        if thismax>4.8:
            xmx=thismax
        if thismin<1.8:
            xmn=thismin
        dm= (dm-xmn)/(xmx-xmn)
        dm*=msk
        im = im[miny : maxy + 1, minx : maxx + 1, :]
        dm = dm[miny : maxy + 1, minx : maxx + 1]

        s = 20
        im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
        dm = np.pad(dm, ((s, s), (s, s)), 'constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        im = im[cy1 : -cy2, cx1 : -cx2, :]
        dm = dm[cy1 : -cy2, cx1 : -cx2]
        return im, dm


    def transform(self, alb, lbl):
        #timestarttransform=time.time()
        alb,lbl=self.tight_crop_d(alb,lbl)               #t,b,l,r = is pixels cropped on top, bottom, left, right
        #print(alb.shape,t,b,l,r)
        #cv2.imwrite('testrecon.png',recon)
        #cv2.imwrite('testimg.png',alb.cpu().numpy())
        #timefinishcrop=time.time()

        #print("crop time: ",timefinishcrop-timestarttransform)
        
        alb = m.imresize(alb, self.img_size)
        #alb=cv2.cvtColor(alb,cv2.COLOR_RGB2BGR)
        alb = alb.astype(float)/255.0 #RGB
        #print(alb.shape)
        alb = alb.transpose(2, 0, 1) # NHWC -> NCHW

        lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
        lbl = np.expand_dims(lbl,axis=0)  # HW -> CHW
        #timefinishtransform=time.time()
        #print("transform: ",timefinishtransform-timestarttransform)

        img = torch.from_numpy(alb).to(torch.float32)
        #print(img.shape)
        lbl = torch.from_numpy(lbl).to(torch.float32)
        #timefinishtorch=time.time()
        #print("np to torch: ",timefinishtorch-timefinishtransform)
        return img, lbl



 
# #Leave code for debugging purposes
if __name__ == '__main__':
    bs = 12
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3dabs_depthfromalb(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True, img_size=(256,256))
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        #print(imgs.shape,labels.shape)
        #print(tobeunwarped.shape)
        #unwarpedimg=F.grid_sample(imgs[:,:,:,:],labels[:,:,:,:])
        #print(labels[0])
        #print(unwarpedimg.shape)
        for i in range(0,imgs.shape[0]):
            cv2.imwrite('testunwarp'+str(i)+'.png',labels[i].permute(1,2,0).cpu().numpy()*255)
#print(prof.key_averages().table(sort_by="self_cpu_time_total"))        
