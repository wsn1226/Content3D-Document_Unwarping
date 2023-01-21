#test end to end benchmark data test
from email.mime import image
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
originalgrid=originalgrid.permute(2,0,1).unsqueeze(0).to(DEVICE)

def gaussian(M, std, sym=True):
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = np.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    kernel_2d=np.outer(w,w)
    return kernel_2d/np.sum(kernel_2d)

weights = torch.from_numpy(gaussian(3,1.5)).to(torch.float)
weights = torch.tensor([[0.0778,	0.1233,	0.0778],
                        [0.1233,	0.1953,	0.1233],
                        [0.0778,	0.1233,	0.0778]])
print(weights)

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
    elif pad=='reflect':
        img=F.pad(img,(1,1,1,1),mode='reflect')
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

    res = F.grid_sample(input=img, grid=(bm))
    #res = res[0].numpy().transpose((1, 2, 0))

    return res


def convertimg(img):
    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).to(torch.float).to(DEVICE)
    return img

def unwarp_fast_nobmsmooth(img, bm):
    img=convertimg(img)
    h,w=img.shape[2],img.shape[3]
    #h,w=1080,1920
    #print(w,h)
    #m = nn.ReplicationPad2d((1, 1, 1, 1))
    #bm=m(bm)
    #bm = bm.permute(0,2,3,1)
    #bm0=cv2.blur(bm[:,:,0],(3,3))
    #bm1=cv2.blur(bm[:,:,1],(3,3))
    #print(bm[:,:1,:,:].shape)
    #bm0=smooth2D(bm[:,:1,:,:],weights,pad='replicate')
    #bm1=smooth2D(bm[:,1:,:,:],weights,pad='replicate')
    #print(bm0.shape)
    #bm0=F.interpolate(bm0,(h,w),mode='bilinear',align_corners=True)
    #bm1=F.interpolate(bm1,(h,w),mode='bilinear',align_corners=True)
    bm=F.interpolate(bm,(h,w),mode='bilinear',align_corners=True)
    #bm=torch.cat([bm0,bm1],dim=1)
    bm=bm.permute(0,2,3,1)

def unwarp_fast(img, bm):
    img=convertimg(img)
    h,w=img.shape[2],img.shape[3]
    #h,w=1080,1920
    #print(w,h)
    #m = nn.ReplicationPad2d((1, 1, 1, 1))
    #bm=m(bm)
    #bm = bm.permute(0,2,3,1)
    #bm0=cv2.blur(bm[:,:,0],(3,3))
    #bm1=cv2.blur(bm[:,:,1],(3,3))
    #print(bm[:,:1,:,:].shape)
    bm0=smooth2D(bm[:,:1,:,:],weights,pad='replicate')
    bm1=smooth2D(bm[:,1:,:,:],weights,pad='replicate')
    #print(bm0.shape)
    bm0=F.interpolate(bm0,(h,w),mode='bilinear',align_corners=True)
    bm1=F.interpolate(bm1,(h,w),mode='bilinear',align_corners=True)
    #bm=F.interpolate(bm,(h,w),mode='bilinear',align_corners=True)
    bm=torch.cat([bm0,bm1],dim=1)
    bm=bm.permute(0,2,3,1)


    res = F.grid_sample(input=img, grid=bm)

    return res

def unwarp_nosmooth(img, bm):
    h,w=img.shape[2],img.shape[3]
    bm=F.interpolate(bm,(h,w),mode='bilinear',align_corners=True)
    bm=bm.permute(0,2,3,1)

    res = F.grid_sample(input=img, grid=bm)

    return res


def test(args,img_path,fname,depth_constraint=False):
    originalstart=time.time()
    #depth_model_file_name = os.path.split(args.depth_model_path)[1]
    #depth_model_name = depth_model_file_name[:depth_model_file_name.find('_',2)]
    boundary_model_name='u2net'
    #depth_model_name='mobilevit_xxsandfcn_skip_depthfromalb'
    depth_model_name='u2net_depth'
    #bm_model_file_name = os.path.split(args.bm_model_path)[1]
    #bm_model_name = bm_model_file_name[:bm_model_file_name.find('_')]
    #bm_model_name='mobilevit_sandfcn_skip'
    bm_model_name='mobilevit_sandfcn_skipRGBD'
    #bm_model_name='mobilevit_sandfcn_skipRGBD_6blocks'
    #bm_model_name='mobilevit_sandfcn_skipRGBD_halfflow'
    #bm_model_name='DocTr'
    #bm_model_name='mobilevit_sanddeeplab3'
    #print(bm_model_name)
    boundary_n_classes = 1
    depth_n_classes = 1
    bm_n_classes = 2

    boundary_img_size=(256,256)
    depth_img_size=(256,256)
    bm_img_size=(128,128)
    # Setup image
    print("Read Input Image from : {}".format(img_path))
    imgorg = cv2.imread(img_path) #BGR
    #imgorg=cv2.resize(imgorg,(1920,1080))
    h,w,_=imgorg.shape
    imgorgbgr=cv2.resize(imgorg, depth_img_size)
    imgbgr = np.expand_dims(imgorgbgr, 0)/255.0
    imgbgr = torch.from_numpy(imgbgr).float().permute(0,3,1,2)

    imgorgrgb = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    imgrgb = cv2.resize(imgorgrgb, depth_img_size)
    #img = img[:, :, ::-1]
    imgrgb = imgrgb.astype(float) / 255.0
    imgrgb = imgrgb.transpose((2, 0, 1)) #CHW
    imgcanbeinputtobm_torch=torch.from_numpy(imgrgb).float().unsqueeze(0).to(DEVICE) #1 3 256 256

    # Predict
    activation=nn.Sigmoid()
    depth_model = get_model(depth_model_name, depth_n_classes, in_channels=3)
    bm_model = get_model(bm_model_name, bm_n_classes, in_channels=4, img_size=bm_img_size)
    boundary_model = get_model(boundary_model_name, boundary_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        depth_state = convert_state_dict(torch.load(args.depth_model_path, map_location='cpu')['model_state'])
        boundary_state = convert_state_dict(torch.load(args.boundary_model_path, map_location='cpu')['model_state'])
        bm_state = convert_state_dict(torch.load(args.bm_model_path, map_location='cpu')['model_state'])
        #print(bm_state)
    else:
        depth_state = convert_state_dict(torch.load(args.depth_model_path)['model_state'])
        boundary_state = convert_state_dict(torch.load(args.boundary_model_path)['model_state'])
        bm_state = convert_state_dict(torch.load(args.bm_model_path)['model_state'])
        #print(bm_state)

    boundary_model.load_state_dict(boundary_state)
    boundary_model.eval()
    depth_model.load_state_dict(depth_state)
    depth_model.eval()
    bm_model.load_state_dict(bm_state)
    bm_model.eval()

    boundary_model.to(DEVICE)
    depth_model.to(DEVICE)
    bm_model.to(DEVICE)
    images = Variable(imgbgr).to(DEVICE)
    #print(images.shape)
    #imgorg=convertimg(imgorg)
    activation=nn.ReLU()



    with torch.no_grad():
        start=time.time()
        boundary_outputs = boundary_model(images)
        time1=time.time()
        #print(time1-start)
        maxpool=nn.MaxPool2d(3,1,1)
        pred_boundary = boundary_outputs[0]
        #pred_boundary=maxpool(pred_boundary)
        #pred_boundary = smooth2D(pred_boundary,weights,pad='constant') # 1,1,256,256
        #pred_boundary=pred_boundary.squeeze(0).squeeze(0)
        mskorg=(pred_boundary>0.5).to(torch.float32)#1,1,256,256
        mskindex=mskorg.nonzero()
        extracted_img=torch.mul(mskorg.repeat(1,3,1,1),imgcanbeinputtobm_torch) # 1,3,256,256

        pred_depth= activation(depth_model(extracted_img)[0])*mskorg
        #cv2.imwrite("testdmap.png",pred_depth.squeeze(0).squeeze(0).cpu().numpy()*255)
        #maxd=torch.max(pred_depth)
        #pred_depth = smooth2D(pred_depth,weights,pad='constant') #1,1,256,256
        #mind=torch.min(pred_depth[mskindex[:,0],mskindex[:,1],mskindex[:,2],mskindex[:,3]])
        maxd=torch.max(pred_depth[mskindex[:,0],mskindex[:,1],mskindex[:,2],mskindex[:,3]])
        #print(maxd)
        #pred_depth[mskindex[:,0],mskindex[:,1],mskindex[:,2],mskindex[:,3]]=(pred_depth[mskindex[:,0],mskindex[:,1],mskindex[:,2],mskindex[:,3]]-mind)/(maxd-mind)
        #pred_depth=mskorg*((pred_depth+0.675)/(0.675+maxd))
        pred_depth=mskorg*pred_depth
        #pred_depth[mskindex[:,0],mskindex[:,1],mskindex[:,2],mskindex[:,3]]=(pred_depth/maxd)[mskindex[:,0],mskindex[:,1],mskindex[:,2],mskindex[:,3]]+1
        #pred_depth-=1
        #final_msk=(pred_depth>0).to(torch.float32)
        #pred_depth=mskorg*(pred_depth/maxd)
        time2=time.time()
        #print(time2-time1)

        bm_input=torch.cat([extracted_img,pred_depth],dim=1)
        bm_input=F.interpolate(bm_input,bm_img_size,mode='bilinear',align_corners=True)
        #print(bm_input.shape)
        cv2.imwrite("testimg.png",bm_input[0,0:3,:,:].permute(1,2,0).cpu().numpy()*255)
        outp_bm=bm_model(bm_input)#+originalgrid
        #outp_bm[0,:,0,1:]=up
        #print(outp_bm.shape)
    
    # Save the output
    #resizedmsk=F.interpolate(msk_org3.permute(2,0,1).unsqueeze(0),(h,w)).to(DEVICE)
    #print(resizedmsk.shape)
    #imgorg,_=tight_crop_4(imgorg,resizedmsk)
    #print(imgorg.shape)
    if not os.path.exists(args.out_path) and args.show:
        os.makedirs(args.out_path)
    if not os.path.exists(args.aux_out_path) and args.show:
        os.makedirs(args.aux_out_path)
        
    outp=os.path.join(args.out_path,fname)
    #cv2.imwrite(os.path.join(args.aux_out_path,fname),(mskorg*(pred_depth/maxd)).squeeze(0).permute(1,2,0).cpu().numpy()*255)


    uwpred=unwarp_fast(imgorg,outp_bm)
    finish=time.time()
    uwpred = uwpred[0].cpu().numpy().transpose((1, 2, 0))
    #uwpred=F.grid_sample(bm_input,outp_bm.permute(0,2,3,1))[0].permute(1,2,0).detach().cpu().numpy()
    if args.show:
        #print(outp)
        cv2.imwrite(outp,uwpred*255)
    print(finish-originalstart)
    torch.cuda.empty_cache()
    return finish-originalstart

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--boundary_model_path', nargs='?', type=str, default='',
                        help='Path to the saved boundary model')
    parser.add_argument('--depth_model_path', nargs='?', type=str, default='',
                        help='Path to the saved depth model')
    parser.add_argument('--bm_model_path', nargs='?', type=str, default='',
                        help='Path to the saved bm model')
    parser.add_argument('--img_path', nargs='?', type=str, default='/disk2/sinan/crop/',
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default='/disk2/sinan/mymodelresult/rgb_consis_absd',
                        help='Path of the output unwarped image')
    parser.add_argument('--aux_out_path', nargs='?', type=str, default='/disk2/sinan/mymodelresult/halfflow',
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

#CUDA_VISIBLE_DEVICES=0 python inferbmfromRGBD_msk.py --boundary_model_path /home/sinan/DewarpNet-master/checkpoints-u2net-nopre/u2net_132_0.010844790506031037_0.04020660579093357_boundary_best_model.pkl --depth_model_path /home/sinan/DewarpNet-master/checkpoints-u2net_abs_depth_fromalb/u2net_depth_85_0.021168790612023168_0.02818983816039219_depth_best_model.pkl --bm_model_path /home/sinan/DewarpNet-master/checkpoints-rgb_absd_withoutoutlier/mobilevit_sandfcn_skipRGBD_83_9.516354199149645e-05_0.00010155977998769776_rgb_absd_best_model.pkl --show
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


