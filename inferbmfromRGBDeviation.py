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
import torchvision.transforms as T
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

n1=np.arange(-1,1,2/256)
grid_x, grid_y = np.meshgrid(n1, n1)
grid=np.stack([grid_x,grid_y],axis=-1).astype(np.float32)
DEVICE = torch.device('cpu')
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights = torch.tensor([[0.0778,	0.1233,	0.0778],
                        [0.1233,	0.1953,	0.1233],
                        [0.0778,	0.1233,	0.0778]])
weights = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(DEVICE)


def smooth2D(img,weights,pad='None'):
    if pad=='constant':
        img=F.pad(img,(1,1,1,1,0,0,0,0))
    elif pad=='replicate':
        img=F.pad(img,(1,1,1,1),mode='replicate') # 0.4564 10.1820
    elif pad=='reflect':
        img=F.pad(img,(1,1,1,1),mode='reflect')
    return F.conv2d(img, weights)


def convertimg(img):
    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float().to(DEVICE)
    return img

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

def ortho_dst_np(allxyz, abc):
    pointmustthrough=np.repeat(np.array([[0,0,1/abc[-1]]]),allxyz.shape[0],axis=0)
    unit_normal=abc/np.sqrt((np.dot(abc,abc)))
    vfrom_plane=allxyz-pointmustthrough
    dst=np.dot(vfrom_plane,unit_normal)
    #print(dst)
    return dst

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
    #bm_model_name='mobilevit_sandfcn_skipRGBD'
    bm_model_name='mobilevit_sandfcn_skipRGBD_halfflow'
    #bm_model_name='bm0false'
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
    imgbgr = np.expand_dims(imgorgbgr, 0)/255.0 #1,256,256,3 0-1
    imgbgr = torch.from_numpy(imgbgr).float().permute(0,3,1,2)#1,3,256,256 0-1

    imgorgrgb = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    imgrgb = cv2.resize(imgorgrgb, depth_img_size)
    imgrgb = np.expand_dims(imgrgb, 0)/255.0 #1,256,256,3 0-1
    imgrgb = torch.from_numpy(imgrgb).float().permute(0,3,1,2)#1,3,256,256 0-1

    # Predict
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
        boundary_state = convert_state_dict(torch.load(args.boundary_model_path, map_location='cpu')['model_state'])
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
        pred_boundary = boundary_outputs[0]
        #pred_boundary=maxpool(pred_boundary)
        #pred_boundary = smooth2D(pred_boundary,weights,pad='constant') # 1,1,256,256
        #pred_boundary=pred_boundary.squeeze(0).squeeze(0)
        mskorg=(pred_boundary>0.5).to(torch.float32)#1,1,256,256
        mskindex=mskorg.nonzero()
        extracted_img=torch.mul(mskorg.repeat(1,3,1,1),imgrgb) # 1,3,256,256

        pred_depth= activation(depth_model(extracted_img)[0])#1,1,256,256
        #pred_depth = smooth2D(pred_depth,weights,pad='constant') #1,1,256,256
        #mind=torch.min(pred_depth[mskindex[:,0],mskindex[:,1],mskindex[:,2],mskindex[:,3]])
        #maxd=torch.max(pred_depth[mskindex[:,0],mskindex[:,1],mskindex[:,2],mskindex[:,3]])
        #print(maxd)
        #pred_depth[mskindex[:,0],mskindex[:,1],mskindex[:,2],mskindex[:,3]]=(pred_depth[mskindex[:,0],mskindex[:,1],mskindex[:,2],mskindex[:,3]]-mind)/(maxd-mind)
        pred_depth=(mskorg*pred_depth).squeeze(0).squeeze(0)#1,1,256,256
        pred_depth=pred_depth.numpy()
        maxd=np.max(pred_depth[mskindex[:,2],mskindex[:,3]])
        mind=np.min(pred_depth[mskindex[:,2],mskindex[:,3]])
        pred_depth[mskindex[:,2],mskindex[:,3]]=(pred_depth[mskindex[:,2],mskindex[:,3]]-mind)/(maxd-mind)
        xy=grid[mskindex[:,2],mskindex[:,3],:]
        xyz=np.hstack([xy,np.expand_dims(pred_depth[mskindex[:,2],mskindex[:,3]],axis=-1)])
        # ax+by+cz=1
        abc=np.linalg.lstsq(xyz,np.ones(xyz.shape[0]),rcond=None)[0]
        dst=ortho_dst_np(xyz,abc)
        #print(dst)
        maxdst=np.max(np.abs(dst))
        # If Visualization: dmap[mskindex]=np.abs(dst/maxdst)
        pred_depth[mskindex[:,2],mskindex[:,3]]=dst/maxdst

        pred_depth=torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0)
        time2=time.time()
        #print(time2-time1)

        extracted_img=F.interpolate(extracted_img,bm_img_size,mode='bilinear',align_corners=True)
        pred_depth=F.interpolate(pred_depth,bm_img_size,mode='bilinear',align_corners=True)
        #mskorg=F.interpolate(mskorg,bm_img_size,mode='bilinear',align_corners=True)
        bm_input=torch.cat([extracted_img,pred_depth],dim=1)
        outp_bm=bm_model(bm_input)
        print(outp_bm.shape)

    outp=os.path.join(args.out_path,fname)
    uwpred=unwarp_fast(imgorg,outp_bm)
    finish=time.time()
    uwpred = uwpred[0].cpu().numpy().transpose((1, 2, 0))
    #uwpred=F.grid_sample(bm_input,outp_bm.permute(0,2,3,1))[0].permute(1,2,0).detach().cpu().numpy()
    if not os.path.exists(args.out_path) and args.show:
        os.makedirs(args.out_path)
    if args.show:
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

#CUDA_VISIBLE_DEVICES=0 python inferbmfromRGBDeviation.py --boundary_model_path /home/sinan/DewarpNet-master/checkpoints-u2net-nopre/u2net_132_0.010844790506031037_0.04020660579093357_boundary_best_model.pkl --depth_model_path /home/sinan/DewarpNet-master/checkpoints-u2depth/u2net_depth_254_0.03333024032177337_0.05067401671253923_depth_best_model.pkl --bm_model_path /home/sinan/DewarpNet-master/checkpoints-rgbd_halfflow/mobilevit_sandfcn_skipRGBD_halfflow_86_0.00016263989673461765_0.0001458648729924448_mobilenetv3_best_model.pkl --show
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


