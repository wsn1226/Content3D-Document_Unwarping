#test end to end benchmark data test
import sys, os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt


from models import get_model
from loaders import get_loader
from utils import convert_state_dict


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE=torch.device('cpu')

weights = torch.tensor([[0.0778,	0.1233,	0.0778],
                        [0.1233,	0.1953,	0.1233],
                        [0.0778,	0.1233,	0.0778]])
weights = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(DEVICE)

def smooth2D(img,weights,pad=False):
    if pad:
        img=F.pad(img,(1,1,1,1,0,0,0,0))
    return F.conv2d(img, weights)

def test(args,img_path,fname):
    depth_model_file_name = os.path.split(args.depth_model_path)[1]
    #depth_model_name = depth_model_file_name[:depth_model_file_name.find('_')]
    #print(depth_model_name)
    depth_model_name='u2net_depth'

    depth_n_classes = 1

    depth_img_size=(256,256)

    # Setup image
    print("Read Input Image from : {}".format(img_path))
    imgorg = cv2.imread(img_path)
    imgorg = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    img = cv2.resize(imgorg, depth_img_size)
    #img = img[:, :, ::-1]
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Predict
    #activation=nn.Sigmoid()
    depth_model = get_model(depth_model_name, depth_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        depth_state = convert_state_dict(torch.load(args.depth_model_path, map_location='cpu')['model_state'])
    else:
        depth_state = convert_state_dict(torch.load(args.depth_model_path)['model_state'])
    depth_model.load_state_dict(depth_state)
    depth_model.eval()

    if torch.cuda.is_available():
        depth_model.to(DEVICE)
        images = Variable(img.to(DEVICE))
    else:
        images = Variable(img)

    with torch.no_grad():
        depth_outputs = depth_model(images)
        pred_depth = depth_outputs[0]

    # Save the output
    outp=os.path.join(args.out_path,fname)
    #pred_depth=smooth2D(pred_depth,weights,pad=True)
    pred_depth=pred_depth.squeeze(0).permute(1,2,0).squeeze(-1)
    msk=(pred_depth>=0).to(torch.float)
    #print(pred_depth[msk.nonzero()[:,0],msk.nonzero()[:,1]].shape)
    #mind=torch.min(pred_depth[msk.nonzero()[:,0],msk.nonzero()[:,1]])
    maxd=torch.max(pred_depth[msk.nonzero()[:,0],msk.nonzero()[:,1]])
    #pred_depth[msk.nonzero()[:,0],msk.nonzero()[:,1]]=(pred_depth[msk.nonzero()[:,0],msk.nonzero()[:,1]]-mind)/(maxd-mind)
    pred_depth[msk.nonzero()[:,0],msk.nonzero()[:,1]]=pred_depth[msk.nonzero()[:,0],msk.nonzero()[:,1]]/maxd
    pred_depth=(msk*pred_depth).cpu().numpy()
    #pred_depth=np.reshape(np.array(pred_depth.permute(0,2,3,1).cpu()),(256,256))
    #pred_depth=np.heaviside(pred_depth-0.5,1)
    cv2.imwrite(outp,pred_depth*255)
    del pred_depth
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--depth_model_path', nargs='?', type=str, default='',
                        help='Path to the saved depth model')
    parser.add_argument('--img_path', nargs='?', type=str, default='/disk2/sinan/crop/',
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default='/disk2/sinan/newdepth_result',
                        help='Path of the output unwarped image')
    parser.add_argument('--show', dest='show', action='store_true',
                        help='Show the input image and output unwarped')
    parser.set_defaults(show=False)
    args = parser.parse_args()
    for fname in os.listdir(args.img_path):
        if '.jpg' in fname or '.JPG' in fname or '.png' in fname:
            img_path=os.path.join( args.img_path,fname)
            test(args,img_path,fname)


# CUDA_VISIBLE_DEVICES=1 python inferdepth.py --depth_model_path /home/sinan/DewarpNet-master/checkpoints-u2depth/u2net_depth_137_0.02782094516774843_0.03947804733937563_depth_best_model.pkl --show
