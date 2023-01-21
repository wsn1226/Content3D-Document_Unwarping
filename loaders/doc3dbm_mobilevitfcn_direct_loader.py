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

class doc3dbm_mobilevitfcn_direct_loader(data.Dataset):
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
        alb_path = pjoin('/home/sinan/DewarpNet-master', 'alb' , im_name + '.png')
        #img_foldr,fname=im_name.split('/')
        img_foldr,fname=im_name.split('/')
        recon_foldr='chess48'
        #recon_foldr='chess48'
        bm_path = pjoin('/home/sinan/DewarpNet-master', 'bm_np' , im_name + '.npy')
        #print(alb_path)
        #print(bm_path)
        recon_path = pjoin(self.root,'recon' , img_foldr, fname[:-4]+recon_foldr+'0001.png')

        #timetostartIO=time.time()

        #alb = m.imread(alb_path,mode='RGB')
        alb=cv2.imread(alb_path)[:,:,::-1]/255.0
        #timefinishalb=time.time()
        #print("alb: ",timefinishalb-timetostartIO)
        bm = np.load(bm_path)
        #recon = m.imread(recon_path,mode='RGB')
        recon = cv2.imread(recon_path)

        #timefinishedIO=time.time()
        #print("bm: ",timefinishedIO-timefinishalb)
        #alb = m.imread(alb_path,mode='RGB')
        #bm = h5.loadmat(bm_path)['bm']

        #timetostarttrans=time.time()       
        if self.is_transform:
            im, lbl = self.transform(alb,bm,recon)
        #timetofinishtrans=time.time()

        #print("IO time:",timefinishedIO-timetostartIO)         
        #print("Preprocessing time:", timetofinishtrans-timetostarttrans) 
        return im, lbl
    


    def tight_crop(self, alb ,recon):
        #msk=((recon[:,:,0]!=0)&(recon[:,:,1]!=0)&(recon[:,:,2]!=0)).astype(np.uint8)
        msk=((alb[:,:,0]!=0)&(alb[:,:,1]!=0)&(alb[:,:,2]!=0)).astype(np.uint8)
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
        #print("albshape",alb.shape)
        recon = recon[miny : maxy + 1, minx : maxx + 1, :]
        
        s = 20
        alb = np.pad(alb, ((s, s), (s, s), (0, 0)), mode='constant')
        recon = np.pad(recon, ((s, s), (s, s), (0, 0)), mode='constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        alb = alb[cy1 : -cy2, cx1 : -cx2, :]
        recon = recon[cy1 : -cy2, cx1 : -cx2, :]
        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2

        return alb,recon,t,b,l,r


    def transform(self, alb, bm, recon):
        #timestarttransform=time.time()
        alb,recon,t,b,l,r=self.tight_crop(alb,recon)               #t,b,l,r = is pixels cropped on top, bottom, left, right
        #print(alb.shape,t,b,l,r)
        #cv2.imwrite('testrecon.png',recon)
        #cv2.imwrite('testimg.png',alb.cpu().numpy())
        #timefinishcrop=time.time()

        #print("crop time: ",timefinishcrop-timestarttransform)
        
        alb = m.imresize(alb, self.img_size)
        alb = alb.astype(float)/255.0
        alb = alb.transpose(2, 0, 1) # NHWC -> NCHW

        recon = m.imresize(recon, self.img_size)
        recon = recon.astype(float)/255.0
        recon = recon.transpose(2, 0, 1) # NHWC -> NCHW

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/np.array([448.0-l-r, 448.0-t-b])
        bm=(bm-0.5)*2

        bm0=cv2.resize(bm[:,:,0],(self.img_size[0],self.img_size[1]))
        bm1=cv2.resize(bm[:,:,1],(self.img_size[0],self.img_size[1]))
        lbl=np.stack([bm0,bm1],axis=-1)
        img=np.concatenate([alb,recon],axis=0)
        #timefinishtransform=time.time()
        #print("transform: ",timefinishtransform-timestarttransform)

        img = torch.from_numpy(img).to(torch.float32)
        lbl = torch.from_numpy(lbl).to(torch.float32)
        #timefinishtorch=time.time()
        #print("np to torch: ",timefinishtorch-timefinishtransform)
        return img, lbl



 
# #Leave code for debugging purposes
if __name__ == '__main__':
    bs = 12
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3dbm_mobilevitfcn_direct_loader(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True, img_size=(128,128))
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        #print(imgs.shape,labels.shape)
        #print(tobeunwarped.shape)
        unwarpedimg=F.grid_sample(imgs[:,:3,:,:],labels[:,:,:,:])
        #print(labels[0])
        #print(unwarpedimg.shape)
        for i in range(0,unwarpedimg.shape[0]):
            cv2.imwrite('testunwarp'+str(i)+'.png',unwarpedimg[i].permute(1,2,0).cpu().numpy()*255)
#print(prof.key_averages().table(sort_by="self_cpu_time_total"))        
