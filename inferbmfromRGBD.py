#test end to end benchmark data test
import sys, os
from sklearn.inspection import partial_dependence
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as transforms
import cv2
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import torchvision
from utils import find4contour,compute_boundary


from models import get_model
from loaders import get_loader
from utils import convert_state_dict

#print(torch.__version__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')
t1=torch.arange(-1,1,2/128)
grid_x, grid_y = torch.meshgrid(t1, t1)
originalgrid=torch.cat([grid_y.unsqueeze(-1),grid_x.unsqueeze(-1)],dim=-1).to(torch.float32)
originalgrid=originalgrid.cuda()

weights = torch.tensor([[0.0778,	0.1233,	0.0778],
                        [0.1233,	0.1953,	0.1233],
                        [0.0778,	0.1233,	0.0778]])
weights = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(DEVICE)

def tight_crop(im, fm):
    # different tight crop
    msk=((fm[:,:,0]!=0)&(fm[:,:,1]!=0)&(fm[:,:,2]!=0)).to(torch.int8)
    #print((msk).nonzero())
    y, x = (msk).nonzero()[:,0],(msk).nonzero()[:,1]
    minx = torch.min(x)
    maxx = torch.max(x)
    miny = torch.min(y)
    maxy = torch.max(y)
    im = im[miny : maxy + 1, minx : maxx + 1, :]
    fm = fm[miny : maxy + 1, minx : maxx + 1, :]
    im=im*fm
    return im, fm

def tight_crop_4(im, fm):
    # different tight crop
    print(fm.shape)
    msk=((fm[:,0,:,:]!=0)&(fm[:,1,:,:]!=0)&(fm[:,2,:,:]!=0)).to(torch.int8)
    print((msk).nonzero())
    y, x = (msk).nonzero()[:,1],(msk).nonzero()[:,2]
    minx = torch.min(x)
    maxx = torch.max(x)
    miny = torch.min(y)
    maxy = torch.max(y)
    im = im[:,:,miny : maxy + 1, minx : maxx + 1]
    #fm = fm[:,:,miny : maxy + 1, minx : maxx + 1]
    return im, fm


def smooth2D(img,weights,pad='None'):
    if pad=='constant':
        img=F.pad(img,(1,1,1,1,0,0,0,0))
    elif pad=='replicate':
        img=F.pad(img,(1,1,1,1),mode='replicate') # 0.4564 10.1820 
    return F.conv2d(img, weights)

def unwarp(img, bm):
    w,h=img.shape[2],img.shape[3]
    #w,h=1920,1080
    bm=bm.squeeze(0)
    bm = bm.permute(1,2,0).detach().cpu().numpy()
    #print(bm[:,:,0].shape)
    bm0=cv2.blur(bm[:,:,0],(3,3))
    bm1=cv2.blur(bm[:,:,1],(3,3))
    bm0=cv2.resize(bm0,(h,w))
    bm1=cv2.resize(bm1,(h,w))
    bm=np.stack([bm0,bm1],axis=-1)
    bm=np.expand_dims(bm,0)
    bm=torch.from_numpy(bm).double()

    res = F.grid_sample(input=img, grid=bm)
    #res = res[0].numpy().transpose((1, 2, 0))

    return res


def convertimg(img):
    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float().to(DEVICE)
    return img

def unwarp_fast(img, bm):
    h,w=img.shape[2],img.shape[3]
    #h,w=1080,1920
    print(w,h)
    #m = nn.ReplicationPad2d((1, 1, 1, 1))
    #bm=m(bm)
    #bm = bm.permute(0,2,3,1)
    #bm0=cv2.blur(bm[:,:,0],(3,3))
    #bm1=cv2.blur(bm[:,:,1],(3,3))
    #print(bm[:,:1,:,:].shape)
    bm0=smooth2D(bm[:,:1,:,:],weights,pad='replicate')
    bm1=smooth2D(bm[:,1:,:,:],weights,pad='replicate')
    print(bm0.shape)
    bm0=F.interpolate(bm0,(h,w),mode='bilinear',align_corners=True)
    bm1=F.interpolate(bm1,(h,w),mode='bilinear',align_corners=True)
    bm=torch.cat([bm0,bm1],dim=1)
    bm=bm.permute(0,2,3,1)

    res = F.grid_sample(input=img, grid=bm)

    return res


def test(args,img_path,fname,depth_constraint=False):
    depth_model_file_name = os.path.split(args.depth_model_path)[1]
    depth_model_name = depth_model_file_name[:depth_model_file_name.find('_',2)]
    depth_model_name='u2net_depth'

    bm_model_file_name = os.path.split(args.bm_model_path)[1]
    #bm_model_name = bm_model_file_name[:bm_model_file_name.find('_')]
    #bm_model_name='mobilevit_sandfcn_skip'
    bm_model_name='mobilevit_sandfcn_skipRGBD'
    #bm_model_name='mobilevit_sandfcn'
    #bm_model_name='bm0false'
    #bm_model_name='mobilevit_sanddeeplab3'
    print(bm_model_name)
    
    depth_n_classes = 1
    bm_n_classes = 2

    depth_img_size=(256,256)
    bm_img_size=(128,128)
    # Setup image
    print("Read Input Image from : {}".format(img_path))
    imgorg = cv2.imread(img_path)
    h,w,_=imgorg.shape
    imgorg = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    imgorgbgr= cv2.cvtColor(imgorg, cv2.COLOR_RGB2BGR)

    imgbgr = cv2.resize(imgorgbgr, depth_img_size)
    #img = img[:, :, ::-1]
    imgbgr = imgbgr.astype(float) / 255.0
    imgbgr = imgbgr.transpose((2, 0, 1))

    imgrgb = cv2.resize(imgorg, depth_img_size)
    #img = img[:, :, ::-1]
    imgrgb = imgrgb.astype(float) / 255.0
    imgrgb = imgrgb.transpose((2, 0, 1))

    imgcanbeinputtobm = imgbgr.transpose((1, 2, 0))


    imgrgb = np.expand_dims(imgrgb, 0)
    imgrgb = torch.from_numpy(imgrgb).float()

    imgcanbeinputtobm_torch=torch.from_numpy(imgcanbeinputtobm.copy()).float().to(DEVICE) #HWC

    # Predict
    activation=nn.Sigmoid()
    depth_model = get_model(depth_model_name, depth_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        depth_state = convert_state_dict(torch.load(args.depth_model_path, map_location='cpu')['model_state'])
    else:
        depth_state = convert_state_dict(torch.load(args.depth_model_path)['model_state'])
    depth_model.load_state_dict(depth_state)
    depth_model.eval()
    bm_model = get_model(bm_model_name, bm_n_classes, in_channels=3, img_size=bm_img_size)
    if DEVICE.type == 'cpu':
        bm_state = convert_state_dict(torch.load(args.bm_model_path, map_location='cpu')['model_state'])
    else:
        bm_state = convert_state_dict(torch.load(args.bm_model_path)['model_state'])
    bm_model.load_state_dict(bm_state)
    bm_model.eval()

    depth_model.to(DEVICE)
    bm_model.to(DEVICE)
    images = Variable(imgrgb).to(DEVICE)
    imgorg=convertimg(imgorg)



    with torch.no_grad():
        start=time.time()
        depth_outputs = depth_model(images)
        pred_depth = depth_outputs[0]
        pred_depth = smooth2D(pred_depth,weights,pad='constant')
        #maxpool=nn.MaxPool2d(3,stride=1,padding=1)
        #pred_depth=maxpool(pred_depth)
        print(pred_depth.shape)
        pred_depth=pred_depth.squeeze(0).squeeze(0)
        mskorg=(pred_depth>=0).to(torch.float32)
        #mind=torch.min(pred_depth[mskorg.nonzero()[:,0],mskorg.nonzero()[:,1]])
        maxd=torch.max(pred_depth[mskorg.nonzero()[:,0],mskorg.nonzero()[:,1]])
        #print(mind,maxd)
        #pred_depth[mskorg.nonzero()[:,0],mskorg.nonzero()[:,1]]=pred_depth[mskorg.nonzero()[:,0],mskorg.nonzero()[:,1]]/maxd # standard 0-1
        pred_depth=(pred_depth+0.675)/(0.675+maxd) # 0.5-1
        #pred_depth=(pred_depth+3)/(3+maxd) # 0.75-1
        pred_depth=mskorg*pred_depth
        msk_org3=mskorg.unsqueeze(-1).expand(256,256,3)

        #bm_input,msk=tight_crop(imgcanbeinputtobm_torch,msk_org3)
        #print(bm_input.shape)
        #bm_input=bm_input.permute(2,0,1)
        #msk=msk.cpu().numpy()
        bm_input=torch.mul(msk_org3,imgcanbeinputtobm_torch).permute(2,0,1)
        #print(torch.max(pred_depth),torch.min(pred_depth))
        bm_input=torch.cat([bm_input,mskorg.unsqueeze(0)],dim=0)
        #bm_input=torch.cat([bm_input,mskorg.unsqueeze(0)],dim=0)
        #print(bm_input.shape)
        bm_input=F.interpolate(bm_input.unsqueeze(0),bm_img_size,mode='bilinear',align_corners=True)
        #print(bm_input.shape) #CHW
        #start=time.time()
        outp_bm=bm_model(bm_input)+originalgrid
        #outp_bm[0,:,0,1:]=up
        #print(outp_bm.shape)
    
    # Save the output
    #resizedmsk=F.interpolate(msk_org3.permute(2,0,1).unsqueeze(0),(h,w)).to(DEVICE)
    #print(resizedmsk.shape)
    #imgorg,_=tight_crop_4(imgorg,resizedmsk)
    #print(imgorg.shape)
    outp=os.path.join(args.out_path,fname)
    if not os.path.exists(args.out_path) and args.show:
        os.makedirs(args.out_path)
    uwpred=unwarp_fast(imgorg,outp_bm)
    finish=time.time()
    uwpred = uwpred[0].cpu().numpy().transpose((1, 2, 0))
    #uwpred=F.grid_sample(bm_input,outp_bm.permute(0,2,3,1))[0].permute(1,2,0).detach().cpu().numpy()
    if args.show:
        print("Here")
        cv2.imwrite(outp,uwpred[:,:,::-1]*255)
    print(finish-start)
    torch.cuda.empty_cache()
    return finish-start

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--depth_model_path', nargs='?', type=str, default='',
                        help='Path to the saved depth model')
    parser.add_argument('--bm_model_path', nargs='?', type=str, default='',
                        help='Path to the saved bm model')
    parser.add_argument('--img_path', nargs='?', type=str, default='/disk2/sinan/crop/',
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default='/disk2/sinan/mymodelresult/halfflow',
                        help='Path of the output unwarped image')
    parser.add_argument('--show', dest='show', action='store_true',
                        help='Show the input image and output unwarped')         
    parser.set_defaults(show=False)
    args = parser.parse_args()
    totaltime=0
    totalnbofimg=0
    for fname in os.listdir(args.img_path):
        if '.jpg' in fname or '.JPG' in fname or '.png' in fname:
            img_path=os.path.join( args.img_path,fname)
            totalnbofimg+=1
            totaltime+=test(args,img_path,fname,depth_constraint=False)
    print(totaltime/totalnbofimg)

# CUDA_VISIBLE_DEVICES=1 python inferbmfromRGBD.py --depth_model_path /home/sinan/DewarpNet-master/checkpoints-u2depth/u2net_depth_137_0.02782094516774843_0.03947804733937563_depth_best_model.pkl --bm_model_path /home/sinan/DewarpNet-master/checkpoints-bm6/mobilevit_sandfcn_skipRGBD_81_0.00012341038450787618_9.10331650976735e-05_bm_6_best_model.pkl --show
""" 
def smooth1D(img, sigma):
    n = int(sigma*(2*torch.log(1000))**0.5)
    x = torch.arange(-n, n+1)
    filter = torch.exp((x**2)/-2/(sigma**2))
    allOneMat = torch.ones(img.shape[0:2])
    resultImg1 = nn.Conv1d(img, filter, 1, torch.float32, 'constant', 0, 0)
    resultAllOne = nn.Conv1d(allOneMat, filter, 1,
                              torch.float32, 'constant', 0, 0)
    img_smoothed = resultImg1/resultAllOne

    return img_smoothed


def smooth2D(img, sigma):
    smoothed1D = smooth1D(img, sigma)
    img_smoothed = (smooth1D(smoothed1D.T, sigma)).T
    return img_smoothed """


