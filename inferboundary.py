#test end to end benchmark data test
import sys, os
from cv2 import resize
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

weights_1 = torch.tensor([[0.0778,	0.1233,	0.0778],
                        [0.1233,	0.1953,	0.1233],
                        [0.0778,	0.1233,	0.0778]])
                        
weights_1 = weights_1.view(1, 1, 3, 3).repeat(1, 1, 1, 1).to(DEVICE)

def smooth2D(img,weights,pad='constant'):
    if pad=='constant':
        img=F.pad(img,(1,1,1,1,0,0,0,0))
    elif pad=='same':
        img=F.pad(img,(1,1,1,1),mode='replicate')
    return F.conv2d(img, weights)

def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        #pretrained_dict = torch.load(path, map_location='cuda:0')
        pretrained_dict = torch.load(path, map_location='cpu')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
        
def test(args,img_path,fname):
    mode='doctr'
    #mode='me'
    boundary_model_file_name = os.path.split(args.boundary_model_path)[1]
    #boundary_model_name = boundary_model_file_name[:boundary_model_file_name.find('_')]
    #boundary_model_name='u2net'
    boundary_model_name='doctr_seg'
    boundary_n_classes = 1

    boundary_img_size=(256,256)

    # Setup image
    print("Read Input Image from : {}".format(img_path))
    imgorg = cv2.imread(img_path)
    h,w,c=imgorg.shape
    #imgorg = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    img = cv2.resize(imgorg, boundary_img_size)
    #img = img[:, :, ::-1]
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Predict
    activation=nn.Sigmoid()
    boundary_model = get_model(boundary_model_name, boundary_n_classes, in_channels=3)
    if mode=='me':
        if DEVICE.type == 'cpu':
            boundary_state = convert_state_dict(torch.load(args.boundary_model_path, map_location='cpu')['model_state'])
        else:
            boundary_state = convert_state_dict(torch.load(args.boundary_model_path)['model_state'])
        boundary_model.load_state_dict(boundary_state)
        boundary_model.eval()
    else:
        model_dict = boundary_model.state_dict()
        pretrained_dict = torch.load(args.boundary_model_path, map_location='cuda:0')
        #print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        #print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        boundary_model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        boundary_model.to(DEVICE)
        images = Variable(img.to(DEVICE))
    else:
        images = Variable(img)

    with torch.no_grad():
        boundary_outputs = boundary_model(images)
        pred_boundary = boundary_outputs[0]
        #pred_boundary = smooth2D(pred_boundary,weights_1,pad='constant')

    # Save the output
    outp=os.path.join(args.out_path,fname)
    if not os.path.exists(args.out_path) and args.show:
        os.makedirs(args.out_path)
    pred_boundary=np.reshape(np.array(pred_boundary.permute(0,2,3,1).cpu()),boundary_img_size)
    pred_boundary=np.heaviside(pred_boundary-0.5,1)
    resized_boundary=cv2.resize(pred_boundary,(w,h)).astype(np.uint8)
    #mask=(resized_boundary!=0).astype(np.uint8)
    #extract_img=imgorg*(np.repeat([resized_boundary],3, axis=-1))
    extract_img=cv2.bitwise_and(imgorg,imgorg,mask=resized_boundary)
    #cv2.imwrite(outp,pred_boundary*255)
    cv2.imwrite(outp,extract_img)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--boundary_model_path', nargs='?', type=str, default='',
                        help='Path to the saved boundary model')
    parser.add_argument('--img_path', nargs='?', type=str, default='/disk2/sinan/crop/',
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default='/disk2/sinan/doctr_seg',
                        help='Path of the output unwarped image')
    parser.add_argument('--show', dest='show', action='store_true',
                        help='Show the input image and output unwarped')
    parser.set_defaults(show=False)
    args = parser.parse_args()
    for fname in os.listdir(args.img_path):
        if '.jpg' in fname or '.JPG' in fname or '.png' in fname:
            img_path=os.path.join( args.img_path,fname)
            test(args,img_path,fname)


#CUDA_VISIBLE_DEVICES=1 python inferboundary.py --boundary_model_path /home/sinan/DewarpNet-master/checkpoints-u2net-nopre/seg.pth --show
