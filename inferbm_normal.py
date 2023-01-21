#test end to end benchmark data test
import sys, os
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


from models import get_model
from loaders import get_loader
from utils import convert_state_dict

#print(torch.__version__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cpu')

weights = torch.tensor([[0.0778,	0.1233,	0.0778],
                        [0.1233,	0.1953,	0.1233],
                        [0.0778,	0.1233,	0.0778]])
weights = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(DEVICE)


def smooth2D(img,weights):
    return F.conv2d(img, weights)

def unwarp(img, bm):
    w,h=img.shape[2],img.shape[3]
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
    print(w,h)
    #bm = bm.permute(0,2,3,1)
    #bm0=cv2.blur(bm[:,:,0],(3,3))
    #bm1=cv2.blur(bm[:,:,1],(3,3))
    #print(bm[:,:1,:,:].shape)
    bm0=smooth2D(bm[:,:1,:,:],weights)
    bm1=smooth2D(bm[:,1:,:,:],weights)
    #print(bm0.shape)
    bm0=F.interpolate(bm0,(h,w),mode='bilinear')
    bm1=F.interpolate(bm1,(h,w),mode='bilinear')
    bm=torch.cat([bm0,bm1],dim=1)
    bm=bm.permute(0,2,3,1)

    res = F.grid_sample(input=img, grid=bm)

    return res


def test(args,img_path,fname):
    nm_path='/disk2/sinan/nmfromboundaryresult/boundaryandnm'+fname
    boundary_model_file_name = os.path.split(args.boundary_model_path)[1]
    boundary_model_name = boundary_model_file_name[:boundary_model_file_name.find('_',2)]
    #print(boundary_model_name)

    bm_model_file_name = os.path.split(args.bm_model_path)[1]
    #bm_model_name = bm_model_file_name[:bm_model_file_name.find('_')]
    bm_model_name='mobilevit_sandfcn_skip'
    #bm_model_name='mobilevit_sandfcn'
    #bm_model_name='bm0false'
    #bm_model_name='mobilevit_sanddeeplab3'
    print(bm_model_name)
    
    boundary_n_classes = 1
    bm_n_classes = 2

    boundary_img_size=(256,256)
    bm_img_size=(128,128)
    # Setup image
    print("Read Input Image from : {}".format(img_path))
    imgorg = cv2.imread(img_path)
    imgorg = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    img = cv2.resize(imgorg, boundary_img_size)
    #img = img[:, :, ::-1]
    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    imgcanbeinputtobm = img.transpose((1, 2, 0))



    print("read normal from : {}".format(nm_path))
    nmorg = cv2.imread(nm_path)
    nmorg = cv2.cvtColor(nmorg, cv2.COLOR_BGR2RGB)
    nm = cv2.resize(imgorg, boundary_img_size)
    #img = img[:, :, ::-1]
    nm = nm.astype(float) / 255.0
    nm=cv2.resize(nm,(128,128))
    nm = nm.transpose((2, 0, 1))
    nm=torch.from_numpy(nm.copy()).float().unsqueeze(0).to(DEVICE)
    #nmcanbeinputtobm = nm.transpose((1, 2, 0))


    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    imgcanbeinputtobm_torch=torch.from_numpy(imgcanbeinputtobm.copy()).float().to(DEVICE) #HWC

    # Predict
    activation=nn.Sigmoid()
    boundary_model = get_model(boundary_model_name, boundary_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        boundary_state = convert_state_dict(torch.load(args.boundary_model_path, map_location='cpu')['model_state'])
    else:
        boundary_state = convert_state_dict(torch.load(args.boundary_model_path)['model_state'])
    boundary_model.load_state_dict(boundary_state)
    boundary_model.eval()
    bm_model = get_model(bm_model_name, bm_n_classes, in_channels=3, img_size=bm_img_size)
    if DEVICE.type == 'cpu':
        bm_state = convert_state_dict(torch.load(args.bm_model_path, map_location='cpu')['model_state'])
    else:
        bm_state = convert_state_dict(torch.load(args.bm_model_path)['model_state'])
    bm_model.load_state_dict(bm_state)
    bm_model.eval()

    boundary_model.to(DEVICE)
    bm_model.to(DEVICE)
    images = Variable(img).to(DEVICE)
    imgorg=convertimg(imgorg)

    with torch.no_grad():
        #start=time.time()
        boundary_outputs = boundary_model(images)
        pred_boundary = boundary_outputs[0]
        pred_boundary=pred_boundary.squeeze(0).squeeze(0)
        msk=(pred_boundary>=0.5).to(torch.float32)
        #msk=msk.cpu().numpy()
        msk=msk.unsqueeze(-1).expand(256,256,3)
        #print(msk.shape)
        #imgcanbeinputtobm=imgcanbeinputtobm_torch.permute()
        #bm_input=torch.from_numpy(cv2.bitwise_and(imgcanbeinputtobm,imgcanbeinputtobm,mask=msk)).float().to(DEVICE).permute(2,0,1) #CHW
        bm_input=torch.mul(msk,imgcanbeinputtobm_torch).permute(2,0,1)
        #print(bm_input.shape)
        bm_input=F.interpolate(bm_input.unsqueeze(0),bm_img_size)
        #print(bm_input.shape) #CHW
        start=time.time()
        #outp_bm=bm_model(bm_input)
        outp_bm=bm_model(nm)
        #print(outp_bm.shape)
    
    # Save the output
    outp=os.path.join(args.out_path,fname)
    uwpred=unwarp_fast(imgorg,outp_bm)
    finish=time.time()
    uwpred = uwpred[0].cpu().numpy().transpose((1, 2, 0))
    #uwpred=F.grid_sample(bm_input,outp_bm.permute(0,2,3,1))[0].permute(1,2,0).detach().cpu().numpy()
    cv2.imwrite(outp,uwpred[:,:,::-1]*255)
    print(finish-start)
    return finish-start

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--boundary_model_path', nargs='?', type=str, default='',
                        help='Path to the saved boundary model')
    parser.add_argument('--bm_model_path', nargs='?', type=str, default='',
                        help='Path to the saved bm model')
    parser.add_argument('--img_path', nargs='?', type=str, default='/disk2/sinan/crop/test',
                        help='Path of the input image')  
    parser.add_argument('--out_path', nargs='?', type=str, default='/disk2/sinan/mymodelresult/bm4result',
                        help='Path of the output unwarped image')
    parser.add_argument('--show', dest='show', action='store_true',
                        help='Show the input image and output unwarped')
    parser.set_defaults(show=False)
    args = parser.parse_args()
    totaltime=0
    for fname in os.listdir(args.img_path):
        if '.jpg' in fname or '.JPG' in fname or '.png' in fname:
            img_path=os.path.join( args.img_path,fname)
            totaltime+=test(args,img_path,fname)
    print(totaltime/6)

# CUDA_VISIBLE_DEVICES=2 python inferbm_normal.py --boundary_model_path ./eval/models/u2net_boundary_best_model.pkl --bm_model_path /home/sinan/DewarpNet-master/checkpoints-bm4/mobilevit_sandfcn_skip_5_0.0003751731041120365_0.0003946773418596059_bm_4_best_model.pkl
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