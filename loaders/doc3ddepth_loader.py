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

from .augmentationsk import data_aug_full_joint, tight_crop, tight_crop_depth_joint_masked_depth

class doc3ddepthLoader(data.Dataset):
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
        im_path = pjoin(self.root, 'img_np',  im_name + '.npy')  
        lbl_path=pjoin('/home/sinan/DewarpNet-master', 'dmap', im_name + '.exr')
        #im = cv2.imread(im_path)[:,:,::-1]
        #im = np.array(im, dtype=np.uint8)
        im=np.load(im_path)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_ANYDEPTH)
        if 'val' in self.split:
            im, lbl=tight_crop_depth_joint_masked_depth(im/255.0,lbl)
        if self.augmentations:          #this is for training, default false for validation\
            tex_id=random.randint(0,len(self.txpths)-1)
            txpth=self.txpths[tex_id]
            #print(self.root[:-7])
            middle_path='doc3d/dtd/images_np/'
            #print(os.path.join(self.root[:-7],middle_path,txpth))
            bg=np.load(os.path.join(self.root[:-7],middle_path,txpth))
            #bg=cv2.resize(tex,self.img_size,interpolation=cv2.INTER_NEAREST)
            im,lbl=data_aug_full_joint(im,lbl,bg)
            #print((lbl>2).astype(np.uint8).nonzero())
            #print((lbl<0).astype(np.uint8).nonzero())
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


    def transform(self, img, lbl):
        img = m.imresize(img, self.img_size) # uint8 with RGB mode
        # plt.imshow(img)
        # plt.show()
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1) # HWC -> CHW

        lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
        lbl = np.expand_dims(lbl,axis=0)  # HW -> CHW

        # to torch
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()

        return img, lbl



if __name__ == '__main__':
    bs = 10
#with torch.autograd.profiler.profile(enabled=True) as prof:
    dst = doc3ddepthLoader(root='/disk2/sinan/doc3d/', split='trainmini', is_transform=True, img_size=(256,256),augmentations=True)
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        #print(imgs.shape)
        #print(labels.shape)
        imgs=imgs.permute(0,2,3,1).numpy()
        labels=labels.permute(0,2,3,1).numpy()
        #print(1-(labels[0]<0).astype(np.float))
        #print((labels<0).astype(np.uint8).nonzero())
        #print(imgs)
        for i in range(imgs.shape[0]):
            cv2.imwrite('test'+str(i)+'.png',imgs[i]*255)

""" msk=(labels!=-1).astype(np.uint8)
labels=(labels-1)*msk """
