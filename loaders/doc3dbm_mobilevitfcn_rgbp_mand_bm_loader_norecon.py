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

t1=torch.arange(-1,1,2/128)
grid_x, grid_y = torch.meshgrid(t1, t1)
originalgrid=torch.cat([grid_y.unsqueeze(-1),grid_x.unsqueeze(-1)],dim=-1).to(torch.float32)

npgridx=grid_x.numpy()
npgridy=grid_y.numpy()

class doc3dbm_mobilevitfcn_rgbp_mand_bm_loader_norecon(data.Dataset):
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
        #recon_foldr='chess48'
        bm_path = pjoin('/home/sinan/DewarpNet-master', 'bm_np' , im_name + '.npy')
        #print(alb_path)
        #print(bm_path)
        #recon_path = pjoin(self.root,'recon' , img_foldr, fname[:-4]+recon_foldr+'0001.png')

        #timetostartIO=time.time()

        #alb = m.imread(alb_path,mode='RGB')
        alb=cv2.imread(alb_path)[:,:,::-1]/255.0
        #timefinishalb=time.time()
        #print("alb: ",timefinishalb-timetostartIO)
        bm = np.load(bm_path)
        #recon = m.imread(recon_path,mode='RGB')
        #recon = m.imread(recon_path)[:,:,::-1]/255

        #timefinishedIO=time.time()
        #print("bm: ",timefinishedIO-timefinishalb)
        #alb = m.imread(alb_path,mode='RGB')
        #bm = h5.loadmat(bm_path)['bm']

        #timetostarttrans=time.time()       
        if self.is_transform:
            rgbp, lbl = self.transform(alb,bm)
        #timetofinishtrans=time.time()

        #print("IO time:",timefinishedIO-timetostartIO)         
        #print("Preprocessing time:", timetofinishtrans-timetostarttrans) 
        return rgbp, lbl
    
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
        #im = np.clip(im, 0., 1.)
        return im

    def tight_crop(self, alb):
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
        #recon = recon[miny : maxy + 1, minx : maxx + 1, :]
        
        s = 20
        alb = np.pad(alb, ((s, s), (s, s), (0, 0)), mode='constant')
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
        msk_alb=((alb[:,:,0]!=0)&(alb[:,:,1]!=0)&(alb[:,:,2]!=0)).astype(np.uint8)
        if 'train' in self.split:
            alb=self.color_jitter(alb,0.2,0.2,0.6,0.6)
            alb*=np.repeat(np.expand_dims(msk_alb,-1),3,axis=-1)
        return alb,t,b,l,r


    def transform(self, alb, bm):
        #timestarttransform=time.time()
        alb,t,b,l,r=self.tight_crop(alb)               #t,b,l,r = is pixels cropped on top, bottom, left, right
        #print(alb.shape,t,b,l,r)
        #cv2.imwrite('testrecon.png',recon)
        #cv2.imwrite('testimg.png',alb.cpu().numpy())
        #timefinishcrop=time.time()

        #print("crop time: ",timefinishcrop-timestarttransform)
        
        alb = m.imresize(alb, self.img_size)
        alb = alb.astype(float)/255.0
        alb = alb.transpose(2, 0, 1) # HWC -> CHW
        #mskalb=((alb[0,:,:]!=0)&(alb[1,:,:]!=0)&(alb[2,:,:]!=0)).astype(np.uint8)

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/np.array([449.0-l-r, 449.0-t-b])
        bm=(bm-0.5)*2

        bm0=cv2.resize(bm[:,:,0],(self.img_size[0],self.img_size[1]))
        bm1=cv2.resize(bm[:,:,1],(self.img_size[0],self.img_size[1]))
        lbl=np.stack([bm0,bm1],axis=-1)

        pmap0=np.abs(bm0-npgridy)
        pmap1=np.abs(bm1-npgridx)
        pmap=pmap0+pmap1
        maxp=np.max(pmap)
        pmap=pmap/maxp
        #timefinishtransform=time.time()
        #print("transform: ",timefinishtransform-timestarttransform)

        img = torch.from_numpy(alb).to(torch.float32)
        pmap = torch.from_numpy(pmap).to(torch.float32).unsqueeze(0)
        rgbp = torch.cat([img,pmap],dim=0)
        lbl = torch.from_numpy(lbl).to(torch.float32)-originalgrid
        #timefinishtorch=time.time()
        #print("np to torch: ",timefinishtorch-timefinishtransform)
        return rgbp, lbl



 
# #Leave code for debugging purposes
if __name__ == '__main__':
    bs = 200
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3dbm_mobilevitfcn_rgbp_mand_bm_loader_norecon(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True, img_size=(128,128))
    trainloader = data.DataLoader(dst, batch_size=bs)
    maxl=-999
    minl=999
    for i, data in enumerate(trainloader):
        imgs, labels = data

        #timestartcuda=time.time()
        #imgs=imgs.cuda()
        #labels=labels.cuda()
        #timefinishcuda=time.time()
        #print("Cuda time: ",timefinishcuda-timestartcuda)
        #print(imgs.shape,labels.shape)
        #print(tobeunwarped.shape)
        #originalgrid=originalgrid.expand(imgs.shape[0],128,128,2).cuda()
        #unwarpedimg=F.grid_sample(imgs[:,:,:,:],labels[:,:,:,:]+originalgrid)
        #print(labels[0])
        #print(unwarpedimg.shape)
        for i in range(0,imgs.shape[0]):
            cv2.imwrite('testp'+str(i)+'.png',imgs[i,3:,:,:].permute(1,2,0).cpu().numpy()*255)
            cv2.imwrite('testimg'+str(i)+'.png',imgs[i,:3,:,:].permute(1,2,0).cpu().numpy()*255)
#print(prof.key_averages().table(sort_by="self_cpu_time_total"))        
