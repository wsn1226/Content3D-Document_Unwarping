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

from tqdm import tqdm
from torch.utils import data

from .augmentationsk import data_aug_boundary, tight_crop_boundary

class doc3dboundaryLoader(data.Dataset):
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
        #self.setup_annotations()
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
        im_path = pjoin(self.root, 'img_np',  im_name + '.npy')  
        #lbl_path=pjoin('/home/sinan/DewarpNet-master', 'boundary_np', im_name + '.npy')
        d_path = pjoin('/home/sinan/DewarpNet-master', 'dmap' , im_name + '.exr')
        #im = m.imread(im_path,mode='RGB')
        #im = np.array(im, dtype=np.uint8)
        im=np.load(im_path)
        #im=cv2.resize(im,(self.img_size))
        #lbl = cv2.imread(lbl_path, -1)
        #lbl = np.array(lbl, dtype=np.float)
        #lbl=np.load(lbl_path)
        dmap=cv2.imread(d_path,cv2.IMREAD_ANYDEPTH)
        lbl=(dmap<1000).astype(np.float)
        if 'val' in self.split:
            im, lbl=tight_crop_boundary(im/255.0,lbl)
        #cv2.imwrite('test.png',im*255)
        #timestartaug=time.time()
        if self.augmentations:          #this is for training, default false for validation\
            tex_id=random.randint(0,len(self.txpths)-1)
            txpth=self.txpths[tex_id] 
            middle_path='/home/sinan/DewarpNet-master/dtd/images_np'
            bg=np.load(os.path.join(middle_path,txpth))
            im,lbl=data_aug_boundary(im,lbl,bg)
        #timestarttransform=time.time()
        #print("aug time: ",timestarttransform-timestartaug)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        #timefinishtransform=time.time()
        #print("transfome time: ",timefinishtransform-timestarttransform)
        return im, lbl


    def transform(self, img, lbl):
        img = cv2.resize(img, self.img_size) # uint8 with RGB mode
        img = img[:, :, ::-1] # RGB -> BGR
        # plt.imshow(img)
        # plt.show()
        img = img.astype(float)
        img = img.transpose(2, 0, 1) # NHWC -> NCHW
        lbl = lbl.astype(float)
        #print(img)
        #print(lbl)
        #normalize label
        lbl = cv2.resize(lbl, self.img_size)
        lbl=np.expand_dims(lbl,-1)
        lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
        lbl = np.array(lbl, dtype=np.float)

        # to torch
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()

        return img, lbl



if __name__ == '__main__':
    bs = 200
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3dboundaryLoader(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True, img_size=(256,256),augmentations=True)
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        #print(imgs.shape,labels.shape)
        imgs,labels=imgs.permute(0,2,3,1).numpy(),labels.permute(0,2,3,1).numpy()

        for i in range(labels.shape[0]):
            cv2.imwrite('testimg'+str(i)+'.png',imgs[i]*255)
            cv2.imwrite('testlabel'+str(i)+'.png',labels[i]*255)
        
