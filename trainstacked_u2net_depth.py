# code to train world coord regression from RGB Image
# models are saved in checkpoints-wc/

import sys, os
from this import d
from cv2 import boundingRect
import cv2
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time
import torchvision.models as models
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils import data
from torchvision import utils
from tqdm import tqdm

from models import get_model
from loaders import get_loader
from utils import show_wc_tnsboard,  get_lr

l1_loss=nn.L1Loss()
bce_loss = nn.BCELoss(reduction='mean')
relu=nn.ReLU()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,pred,labels):#NCHW
        if pred.shape!=labels.shape:
            print("Shapes are inconsistent with each other")
        N,C,H,W=pred.shape
        n=H*W
        d=torch.log(pred)-torch.log(labels)
        return (1/n*(torch.sum(d**2))-(1/n**2)*(torch.sum(d))**2)/N

def muti_l1_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = l1_loss(d0,labels_v)
	loss1 = l1_loss(d1,labels_v)
	loss2 = l1_loss(d2,labels_v)
	loss3 = l1_loss(d3,labels_v)
	loss4 = l1_loss(d4,labels_v)
	loss5 = l1_loss(d5,labels_v)
	loss6 = l1_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	#print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss


def write_log_file(log_file_name,losses, epoch, lrate, phase):
    with open(log_file_name,'a') as f:
        f.write("\n{} LRate: {} Epoch: {} Loss: {}".format(phase, lrate, epoch, losses[0]))


def train(args):

    # Setup Dataloader
    data_loader = get_loader('doc3ddepth')
    data_path = args.data_path
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols),augmentations=True)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=12, shuffle=True,pin_memory=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=12,pin_memory=True)

    # Setup Model

    model_d1 = get_model(args.arch_d1, n_classes=1,in_channels=3, img_size=(args.img_rows,args.img_cols)) #Get the U-Net architecture
    model_d1 = torch.nn.DataParallel(model_d1, device_ids=range(torch.cuda.device_count()))
    model_d1.to(device)

    model_d2 = get_model(args.arch_d2, n_classes=1,in_channels=1, img_size=(args.img_rows,args.img_cols)) #Get U2Net
    model_d2 = torch.nn.DataParallel(model_d2, device_ids=range(torch.cuda.device_count()))
    model_d2.to(device)

    # Activation
    
    # optimizer_d1
    #optimizer_d2= torch.optim.Adam(model_d2.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)
    #optimizer_d1 = torch.optim.Adam(model.parameters(), lr=args.l_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # LR Scheduler, which can reduce the learning rate as time passing by
    #sched_d2=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d2, mode='min', factor=0.5, patience=5, verbose=True)

    # Losses
    MSE = nn.MSELoss() #L2 Loss for measure the training loss and validation loss
    #loss_fn = nn.DepthLoss() # Depth Loss used throughout the whole training process, including L_C, L_D, L_T (Both in the first and second training phase)
    #gloss= grad_loss.Gradloss(window_size=5,padding=2) #The Gradient Loss used in L_C, i.e. deltaChead-deltaC

    epoch_start=0
    if args.resume_d1 is not None:                               
        if os.path.isfile(args.resume_d1):# Loading model and optimizer_d1 from the checkpoint
            print("Loading model and optimizer_d1 from checkpoint '{}'".format(args.resume_d1))
            checkpoint = torch.load(args.resume_d1)
            (model_d1).load_state_dict(checkpoint['model_state'])
            #optimizer_d1.load_state_dict(checkpoint['optimizer_d1_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                .format(args.resume_d1, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume_d1)) 
    if args.resume_d2 is not None:                               
        if os.path.isfile(args.resume_d2):# Loading model and optimizer_d1 from the checkpoint
            print("Loading model and optimizer_d2 from checkpoint '{}'".format(args.resume_d2))
            checkpoint = torch.load(args.resume_d2)
            (model_d2).load_state_dict(checkpoint['model_state'])
            #optimizer_d1.load_state_dict(checkpoint['optimizer_d1_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                .format(args.resume_d2, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume_d2)) 
    model_stacked_u2net=get_model(name='stacked_u2net_depth',d1_model=model_d1, d2_model=model_d2)
    optimizer_stacked= torch.optim.Adam(model_stacked_u2net.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)
    sched_joint=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_stacked, mode='min', factor=0.5, patience=3, verbose=True)
    #optimizer_stacked.load_state_dict(checkpoint['optimizer_state'])
    
    #Log file:
    if not os.path.exists(args.logdir_d1):
        os.makedirs(args.logdir_d1)
    experiment_name='stacked_u2net_depth_d1'
    log_file_name_d1=os.path.join(args.logdir_d1,experiment_name+'.txt')
    if os.path.isfile(log_file_name_d1):
        log_file=open(log_file_name_d1,'a')
    else:
        log_file=open(log_file_name_d1,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    if not os.path.exists(args.logdir_d2):
        os.makedirs(args.logdir_d2)
    experiment_name='stacked_u2net_depth_d2'
    log_file_name_d2=os.path.join(args.logdir_d2,experiment_name+'.txt')
    if os.path.isfile(log_file_name_d2):
        log_file=open(log_file_name_d2,'a')
    else:
        log_file=open(log_file_name_d2,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    # Setup tensorboard for visualization
    if args.tboard:
        # save logs in runs/<experiment_name> 
        writer = SummaryWriter(comment=experiment_name)

    
    best_val_loss_d1 = 99999.0
    best_val_loss_d2 = 99999.0
    global_step=0

    print(get_lr(optimizer_stacked))


    for epoch in range(epoch_start,args.n_epoch):
        avg_loss_d1=0.0
        avg_loss2_d1=0.0
        avg_lossall_d1=0.0

        avg_loss_d2=0.0
        avg_loss2_d2=0.0
        avg_lossall_d2=0.0

        model_d1.train()
        model_d2.train()

        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.to(device))
            dmap = Variable(labels.to(device))
            optimizer_stacked.zero_grad()

            #labels_msk=(dmap>=0).to(torch.float)
            #print(msk_index.shape)

            b1,b2,b3,b4,b5,b6,b7, d0, d1, d2, d3, d4, d5, d6= model_stacked_u2net(images)
            loss2_d1, loss_d1 = muti_l1_loss_fusion(b1,b2,b3,b4,b5,b6,b7, dmap)
            avg_loss2_d1+=float(loss2_d1)
            avg_lossall_d1+=float(loss_d1)
            avg_loss_d1+=float(loss_d1)

            #pred_msk=(b1==0).to(torch.float32)

            loss2_d2, loss_d2 = muti_l1_loss_fusion(d0, d1, d2, d3, d4, d5, d6, dmap)
            avg_loss2_d2+=float(loss2_d2)
            avg_lossall_d2+=float(loss_d2)
            avg_loss_d2+=float(loss_d2)

            loss=loss_d1+loss_d2


            loss.backward()
            optimizer_stacked.step()
            global_step+=1

            if (i+1) % 50 == 0:
                print("Epoch[%d/%d] Batch [%d/%d] D1 Loss: %.4f D2 Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader),avg_loss_d1/50.0, avg_loss_d2/50.0))
                avg_loss_d2=0.0
                avg_loss_d1=0.0

            if args.tboard and  (i+1) % 20 == 0:
                #show_wc_tnsboard(global_step, writer,images,labels,pred, 8,'Train Inputs', 'Train Depths', 'Train Pred. Depths')
                writer.add_scalar('Depth: L1 Loss/train', avg_lossall_d1/(i+1), global_step)
                writer.add_scalar('Depth: L1 Loss/train', avg_lossall_d2/(i+1), global_step)
            del b1,b2,b3,b4,b5,b6,b7,d0, d1, d2, d3, d4, d5, d6, loss_d1,loss_d2
            if (i+1) % 50==0:
                torch.cuda.empty_cache()

        avg_loss2_d1=avg_loss2_d1/len(trainloader)
        avg_loss2_d2=avg_loss2_d2/len(trainloader)
        #avg_gloss=avg_gloss/len(trainloader)
        print("Training d1 loss2:%4f d2 loss2:%4f" %(avg_loss2_d1, avg_loss2_d2))

        train_losses_d1=[avg_loss2_d1]
        lrate_d1=get_lr(optimizer_stacked)
        write_log_file(log_file_name_d1, train_losses_d1, epoch+1, lrate_d1,'Train')

        train_losses_d2=[avg_loss2_d2]
        lrate_d2=get_lr(optimizer_stacked)
        write_log_file(log_file_name_d2, train_losses_d2, epoch+1, lrate_d2,'Train')
        #-----------------EVALUATION-----------------
        model_stacked_u2net.eval()
        
        val_loss_d1=0.0
        val_loss_d2=0.0
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)): #use progress bar
            with torch.no_grad():
                images_val = Variable(images_val.to(device))
                depth_val = Variable(labels_val.to(device))

                depths= model_stacked_u2net(images_val)
                d1,d2=depths[0],depths[7]

                #pred_msk=pred_msk.cpu()
                depth_val=depth_val.cpu()
                d1=d1.cpu()
                d2=d2.cpu()

                loss_d1=l1_loss(d1,depth_val)
                loss_d2=l1_loss(d2,depth_val)

                val_loss_d1+=float(loss_d1)   
                val_loss_d2+=float(loss_d2)          
                
                del depths,d1,d2,depth_val
                if (i_val+1) % 50==0:
                    torch.cuda.empty_cache()

        if args.tboard:
            writer.add_scalar('Depth: L1 Loss/val', val_loss_d2/len(valloader), epoch+1)

        val_loss_d1=val_loss_d1/len(valloader)
        val_loss_d2=val_loss_d2/len(valloader)
        print("val loss at epoch {}:: {}".format(epoch+1,val_loss_d2))

        val_losses_d1=[val_loss_d1]
        write_log_file(log_file_name_d1, val_losses_d1, epoch+1, lrate_d1, 'Val')

        val_losses_d2=[val_loss_d2]
        write_log_file(log_file_name_d2, val_losses_d2, epoch+1, lrate_d2, 'Val')

        #reduce learning rate
        sched_joint.step(val_loss_d2)


        if val_loss_d1<best_val_loss_d1:
            best_val_loss_d1=val_loss_d1

        if val_loss_d2<best_val_loss_d2:
            best_val_loss_d2=val_loss_d2
        state = {'epoch': epoch+1,
                    'model_state': (model_stacked_u2net.d1_model).state_dict(),
                    'optimizer_state' : optimizer_stacked.state_dict(),}
        torch.save(state, args.logdir_d1+"{}_{}_{}_{}_best_model.pkl".format(args.arch_d1, epoch+1,val_loss_d1, experiment_name))
        state = {'epoch': epoch+1,
                    'model_state': (model_stacked_u2net.d2_model).state_dict(),
                    'optimizer_state' : optimizer_stacked.state_dict(),}
        torch.save(state, args.logdir_d2+"{}_{}_{}_{}_best_model.pkl".format(args.arch_d2, epoch+1,val_loss_d2, experiment_name))
        torch.cuda.empty_cache()
        time.sleep(0.003)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch_d1', nargs='?', type=str, default='u2net_depth', 
                        help='Architecture to use')
    parser.add_argument('--arch_d2', nargs='?', type=str, default='u2net_depth', 
                        help='Architecture to use')
    parser.add_argument('--data_path', nargs='?', type=str, default='', 
                        help='Data path to load data')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=400, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=200, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-05, 
                        help='Learning Rate')
    parser.add_argument('--resume_d2', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--resume_d1', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir_d2', nargs='?', type=str, default='./checkpoints-stacked_d2/',    
                        help='Path to store the loss logs')
    parser.add_argument('--logdir_d1', nargs='?', type=str, default='./checkpoints-stacked_d1/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.set_defaults(tboard=True)

    args = parser.parse_args()
    train(args)


#CUDA_VISIBLE_DEVICES=2,0,1 python trainstacked_u2net_depth.py --data_path /disk2/sinan/doc3d/ --batch_size 35 --tboard --resume_d1 /home/sinan/DewarpNet-master/checkpoints-u2depth/u2net_depth_185_0.06452940666261144_0.039535758352571435_depth_best_model.pkl