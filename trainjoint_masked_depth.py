# code to train world coord regression from RGB Image
# models are saved in checkpoints-wc/

import sys, os
from this import d
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
    data_loader = get_loader('doc3djoint_masked_depth')
    data_path = args.data_path
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols),augmentations=True)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=12, shuffle=True,pin_memory=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=12,pin_memory=True)

    # Setup Model

    model_depth = get_model(args.arch_depth, n_classes,in_channels=3, img_size=(args.img_rows,args.img_cols)) #Get U2Net
    model_depth = torch.nn.DataParallel(model_depth, device_ids=[0,1,2])
    model_depth.to(device)
    
    model_bm = get_model(args.arch_bm, n_classes,in_channels=4, img_size=(args.bm_img_rows,args.bm_img_cols)) #Get mobileViT-rgbd
    model_bm = torch.nn.DataParallel(model_bm, device_ids=[0,1,2])
    model_bm.to(device)
    # Activation
    
    # optimizer_boundary
    #optimizer_depth= torch.optim.Adam(model_depth.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)
    #optimizer_boundary = torch.optim.Adam(model.parameters(), lr=args.l_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # LR Scheduler, which can reduce the learning rate as time passing by
    #sched_depth=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_depth, mode='min', factor=0.5, patience=5, verbose=True)

    # Losses
    MSE = nn.MSELoss() #L2 Loss for measure the training loss and validation loss
    #loss_fn = nn.DepthLoss() # Depth Loss used throughout the whole training process, including L_C, L_D, L_T (Both in the first and second training phase)
    #gloss= grad_loss.Gradloss(window_size=5,padding=2) #The Gradient Loss used in L_C, i.e. deltaChead-deltaC

    epoch_start=0
    if args.resume_depth is not None:                               
        if os.path.isfile(args.resume_depth):# Loading model and optimizer_boundary from the checkpoint
            print("Loading model and optimizer_depth from checkpoint '{}'".format(args.resume_depth))
            checkpoint = torch.load(args.resume_depth)
            (model_depth).load_state_dict(checkpoint['model_state'])
            #optimizer_boundary.load_state_dict(checkpoint['optimizer_boundary_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                .format(args.resume_depth, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume_depth)) 
    if args.resume_bm is not None:                               
        if os.path.isfile(args.resume_bm):# Loading model and optimizer_boundary from the checkpoint
            print("Loading model and optimizer_bm from checkpoint '{}'".format(args.resume_bm))
            checkpoint = torch.load(args.resume_bm)
            (model_bm).load_state_dict(checkpoint['model_state'])
            #optimizer_joint.load_state_dict(checkpoint['optimizer_boundary_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                .format(args.resume_bm, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume_bm))
    model_joint=get_model(name='u2netdepth_joint_mobilevit',depth_model=model_depth,bm_model=model_bm)
    optimizer_joint= torch.optim.Adam(model_joint.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)
    sched_joint=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_joint, mode='min', factor=0.5, patience=5, verbose=True)
    
    #Log file:
    if not os.path.exists(args.logdir_depth):
        os.makedirs(args.logdir_depth)
    experiment_name='joint_masked_depth'
    log_file_name_depth=os.path.join(args.logdir_depth,experiment_name+'.txt')
    if os.path.isfile(log_file_name_depth):
        log_file=open(log_file_name_depth,'a')
    else:
        log_file=open(log_file_name_depth,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    if not os.path.exists(args.logdir_bm):
        os.makedirs(args.logdir_bm)
    log_file_name_bm=os.path.join(args.logdir_bm,experiment_name+'.txt')
    if os.path.isfile(log_file_name_bm):
        log_file=open(log_file_name_bm,'a')
    else:
        log_file=open(log_file_name_bm,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    # Setup tensorboard for visualization
    if args.tboard:
        # save logs in runs/<experiment_name> 
        writer = SummaryWriter(comment=experiment_name)

    best_val_loss_depth = 99999.0
    best_val_loss_bm = 99999.0
    global_step=0

    print(get_lr(optimizer_joint))


    for epoch in range(epoch_start,args.n_epoch):

        avg_loss_depth=0.0
        avg_loss2_depth=0.0
        avg_lossall_depth=0.0

        avg_loss_bm=0.0
        avgl1loss_bm=0.0


        model_depth.train()
        model_bm.train()

        for i, (images, dmap, bm) in enumerate(trainloader):
            images = Variable(images.to(device))

            #-------------------DEPTH--------------------------
            dmap = Variable(dmap.to(device))
            optimizer_joint.zero_grad()

            labels_msk=(dmap>=0).to(torch.float)
            msk_index=(labels_msk).nonzero()

            #print(msk_index.shape)

            d0, d1, d2, d3, d4, d5, d6, pred_bm = model_joint(images,labels_msk,(args.bm_img_rows,args.bm_img_cols))

            only_d0=d0[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d1=d1[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d2=d2[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d3=d3[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d4=d4[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d5=d5[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d6=d6[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d_lbl=dmap[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            #print(only_d0)

            loss2_depth, loss_depth = muti_l1_loss_fusion(only_d0, only_d1, only_d2, only_d3, only_d4, only_d5, only_d6, only_d_lbl)

            avg_loss2_depth+=float(loss2_depth)
            avg_lossall_depth+=float(loss_depth)
            avg_loss_depth+=float(loss_depth)

            #-------------------BM--------------------------
            bm = Variable(bm.to(device))
            predbm_nhwc = pred_bm.permute(0,2,3,1)
            loss_bm = l1_loss(predbm_nhwc, bm)

            avgl1loss_bm+=float(loss_bm)        
            avg_loss_bm+=float(loss_bm)

            loss=loss_depth+loss_bm


            loss.backward()
            optimizer_joint.step()
            global_step+=1

            if (i+1) % 50 == 0:
                print("Epoch[%d/%d] Batch [%d/%d] Depth Loss: %.4f BM Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader), avg_loss_depth/50.0, avg_loss_bm/50.0 ))
                avg_loss_depth=0.0
                avg_loss_bm=0.0

            if args.tboard and  (i+1) % 20 == 0:
                #show_wc_tnsboard(global_step, writer,images,labels,pred, 8,'Train Inputs', 'Train Depths', 'Train Pred. Depths')
                writer.add_scalar('Depth: L1 Loss/train', avg_lossall_depth/(i+1), global_step)
                writer.add_scalar('BM: L1 Loss/train', avgl1loss_bm/(i+1), global_step)
            del d0, d1, d2, d3, d4, d5, d6,only_d0, only_d1, only_d2, only_d3, only_d4, only_d5, only_d6, pred_bm
            if (i+1) % 50==0:
                torch.cuda.empty_cache()
                #writer.add_scalar('Depth: Grad Loss/train', avg_gloss/(i+1), global_step)
        #avg_loss=avg_loss/len(trainloader)
        avg_loss2_depth=avg_loss2_depth/len(trainloader)
        avgl1loss_bm=avgl1loss_bm/len(trainloader)
        #avg_gloss=avg_gloss/len(trainloader)
        print("Training depth loss2:%4f BM L1:%4f" %(avg_loss2_depth, avgl1loss_bm))
        train_losses_depth=[avg_loss2_depth]
        lrate_depth=get_lr(optimizer_joint)
        write_log_file(log_file_name_depth, train_losses_depth, epoch+1, lrate_depth,'Train')
        
        train_losses_bm=[avgl1loss_bm]
        lrate_bm=get_lr(optimizer_joint)
        write_log_file(log_file_name_bm, train_losses_bm,epoch+1, lrate_bm,'Train')

        model_joint.eval()
        
        val_loss_depth=0.0
        val_loss_bm=0.0
        for i_val, (images_val, depth_val, bm_val) in tqdm(enumerate(valloader)): #use progress bar
            with torch.no_grad():
                images_val = Variable(images_val.to(device))

                #-------------------DEPTH--------------------------
                depth_val = Variable(depth_val.to(device))

                labels_msk=(depth_val>=0).to(torch.float)
                msk_index=(labels_msk).nonzero()

                d1,d2,d3,d4,d5,d6,d7,pred_bm= model_joint(images_val,labels_msk,(args.bm_img_rows,args.bm_img_cols))

                only_d1=d1[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
                only_d_lbl=depth_val[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]

                loss_depth=l1_loss(only_d1.cpu(),only_d_lbl.cpu())
                val_loss_depth+=float(loss_depth)

                #-------------------BM------------------------
                predbm_nhwc = pred_bm.permute(0,2,3,1)
                l1loss = l1_loss(predbm_nhwc.cpu(), bm_val.cpu())
                #rloss,ssim,uworg,uwpred = reconst_loss(images_val[:,:-1,:,:],target_nhwc,labels_val)
                val_loss_bm+=float(l1loss)              
                
                del d1,d2,d3,d4,d5,d6,d7,only_d_lbl,only_d1,labels_msk,predbm_nhwc,pred_bm
                if (i_val+1) % 50==0:
                    torch.cuda.empty_cache()

        if args.tboard:
            #show_wc_tnsboard(epoch+1, writer,images_val,labels_val,pred, 8,'Val Inputs', 'Val Depths', 'Val Pred. Depths')
            writer.add_scalar('Depth: L1 Loss/val', val_loss_depth/len(valloader), epoch+1)
            writer.add_scalar('BM: L1 Loss/val', val_loss_bm/len(valloader), epoch+1)

        val_loss_depth=val_loss_depth/len(valloader)
        val_loss_bm=val_loss_bm/len(valloader)
        print("Depth val loss at epoch {}:: {}".format(epoch+1,val_loss_depth))
        print("BM val loss at epoch {}:: {}".format(epoch+1,val_loss_bm))

        val_losses_depth=[val_loss_depth]
        write_log_file(log_file_name_depth, val_losses_depth, epoch+1, lrate_depth, 'Val')

        val_losses_bm=[val_loss_bm]
        write_log_file(log_file_name_bm, val_losses_bm, epoch+1, lrate_bm, 'Val')

        #reduce learning rate
        sched_joint.step(val_loss_bm)

        if val_loss_depth<best_val_loss_depth:
            best_val_loss_depth=val_loss_depth
        
        if val_loss_bm < best_val_loss_bm:
            best_val_loss_bm=val_loss_bm
        state = {'epoch': epoch+1,
                    'model_state': (model_joint.depth_model).state_dict(),
                    'optimizer_state' : optimizer_joint.state_dict(),}
        torch.save(state, args.logdir_depth+"{}_{}_{}_{}_best_model.pkl".format(args.arch_depth, epoch+1,val_loss_depth, experiment_name))
        state = {'epoch': epoch+1,
                    'model_state': (model_joint.bm_model).state_dict(),
                    'optimizer_state' : optimizer_joint.state_dict(),}
        torch.save(state, args.logdir_bm+"{}_{}_{}_{}_best_model.pkl".format(args.arch_bm, epoch+1,val_loss_bm,experiment_name))
        torch.cuda.empty_cache()
        time.sleep(0.003)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch_depth', nargs='?', type=str, default='u2net_depth', 
                        help='Architecture to use')
    parser.add_argument('--arch_bm', nargs='?', type=str, default='mobilevit_sandfcn_skipRGBD', 
                        help='Architecture to use')
    parser.add_argument('--data_path', nargs='?', type=str, default='', 
                        help='Data path to load data')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Width of the input image')
    parser.add_argument('--bm_img_rows', nargs='?', type=int, default=128, 
                        help='Height of the input image')
    parser.add_argument('--bm_img_cols', nargs='?', type=int, default=128, 
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=400, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=200, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=0.00001, 
                        help='Learning Rate')
    parser.add_argument('--resume_depth', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--resume_bm', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir_depth', nargs='?', type=str, default='./checkpoints-jointmskdepth-u2depth_no+1/',    
                        help='Path to store the loss logs')
    parser.add_argument('--logdir_bm', nargs='?', type=str, default='./checkpoints-jointmskdepth-bm_no+1/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.set_defaults(tboard=True)

    args = parser.parse_args()
    train(args)


#CUDA_VISIBLE_DEVICES=0,1,2 python trainjoint_masked_depth.py --data_path /disk2/sinan/doc3d/ --batch_size 64 --tboard --resume_bm /home/sinan/DewarpNet-master/checkpoints-jointmskdepth-bm_no+1/mobilevit_sandfcn_skipRGBD_88_0.013960556755922024_joint_masked_depth_best_model.pkl --resume_depth /home/sinan/DewarpNet-master/checkpoints-jointmskdepth-u2depth_no+1/u2net_depth_88_0.05127195043453745_joint_masked_depth_best_model.pkl