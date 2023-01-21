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

from .augmentationsk import data_aug_joint_masked_depth, tight_crop, tight_crop_boundary_joint_masked_depth

def color_jitter(im, brightness=0, contrast=0, saturation=0, hue=0):
    f = random.uniform(1 - contrast, 1 + contrast)
    im = np.clip(im * f, 0., 1.)
    f = random.uniform(-brightness, brightness)
    im = np.clip(im + f, 0., 1.).astype(np.float32)

    # hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    # f = random.uniform(-hue, hue)
    # hsv[0] = np.clip(hsv[0] + f * 360, 0., 360.)
    # f = random.uniform(-saturation, saturation)
    # hsv[2] = np.clip(hsv[2] + f, 0., 1.)
    # im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # im = np.clip(im, 0., 1.)
    return im

class doc3djoint_masked_depth(data.Dataset):
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


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                # 1/824_8-cp_Page_0503-7Nw0001
        #im_path = pjoin(self.root, 'img',  im_name + '.png')  
        alb_path = pjoin('/home/sinan/DewarpNet-master', 'alb' , im_name + '.png')

        #labels 
        d_path = pjoin('/home/sinan/DewarpNet-master', 'dmap' , im_name + '.exr')
        bm_path = pjoin('/home/sinan/DewarpNet-master', 'bm_np' , im_name + '.npy')

        # Load data
        im=cv2.imread(alb_path)[:,:,::-1]/255.0
        dmap=cv2.imread(d_path,cv2.IMREAD_ANYDEPTH)
        bm=np.load(bm_path)

        if self.is_transform:
            im, dmap, bm = self.transform(im, dmap, bm)
        #timefinishtransform=time.time()
        #print("transfome time: ",timefinishtransform-timestarttransform)
        return im, dmap, bm

    def tight_crop_boundary_joint_masked_depth(self, im, dmap):
        # different tight crop
        msk=(dmap<10000).astype(np.uint8)
        size=msk.shape
        dmap=dmap*msk
        [y, x] = (msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        maxd=np.max(dmap[msk.nonzero()])
        mind=np.min(dmap[msk.nonzero()])
        dmap[msk.nonzero()]=(dmap[msk.nonzero()]-mind)/(maxd-mind)+1

        im = im[miny : maxy + 1, minx : maxx + 1, :]
        dmap = dmap[miny : maxy + 1, minx : maxx + 1]
        
        s = 20
        im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
        dmap = np.pad(dmap, ((s, s), (s, s)), 'constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        im = im[cy1 : -cy2, cx1 : -cx2, :]
        dmap= dmap [cy1 : -cy2, cx1 : -cx2]
        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2
        if self.augmentations:
            im=color_jitter(im, 0.2, 0.2, 0.6, 0.6)
        return im,dmap,t,b,l,r

    def transform(self, img, dmap, bm):
        img, dmap,t,b,l,r=self.tight_crop_boundary_joint_masked_depth(img,dmap)

        img = m.imresize(img, self.img_size) # uint8 with RGB mode
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1) # NHWC -> NCHW

        #normalize depth map
        dmap = cv2.resize(dmap, self.img_size, interpolation=cv2.INTER_NEAREST)
        dmap = np.expand_dims(dmap,axis=0)-1  # HW -> CHW

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/np.array([448.0-l-r, 448.0-t-b])
        bm=(bm-0.5)*2

        bm0=cv2.resize(bm[:,:,0],(128,128))
        bm1=cv2.resize(bm[:,:,1],(128,128))
        bm=np.stack([bm0,bm1],axis=-1)

        # to torch
        img = torch.from_numpy(img).to(torch.float)
        dmap = torch.from_numpy(dmap).to(torch.float)
        bm = torch.from_numpy(bm).to(torch.float) #HWC->CHW
        #lbl=torch.cat([boundary,dmap,bm],dim=0)

        return img, dmap, bm



if __name__ == '__main__':
    bs = 12
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3djoint_masked_depth(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True,augmentations=True, img_size=(128,128))
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        img, dmap, bm = data
        #print(imgs.shape,labels.shape)
        #imgs,labels=imgs.permute(0,2,3,1).numpy(),labels.permute(0,2,3,1).numpy()
        unwarpedimg=F.grid_sample(dmap,bm)
        unwarpedimg=unwarpedimg.permute(0,2,3,1).numpy()

        for i in range(img.shape[0]):
            #cv2.imwrite('testimg'+str(i)+'.png',unwarpedimg[i]*255)
            cv2.imwrite('testlabel'+str(i)+'.png',img[i].permute(1,2,0).numpy()*255)
        
