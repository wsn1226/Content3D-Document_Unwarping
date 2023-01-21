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

#torch.autograd.set_detect_anomaly(True)
l1_loss=nn.L1Loss()
cos_loss=nn.CosineEmbeddingLoss()
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

def muti_cos_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    d0 = d0.view(-1,3)
    d1 = d1.view(-1,3)
    d2 = d2.view(-1,3)
    d3 = d3.view(-1,3)
    d4 = d4.view(-1,3)
    d5 = d5.view(-1,3)
    d6 = d6.view(-1,3)
    labels_v = labels_v.view(-1,3)
    obj=torch.ones(d0.shape[0]).cuda()

    loss0 = cos_loss(d0,labels_v,obj)
    loss1 = cos_loss(d1,labels_v,obj)
    loss2 = cos_loss(d2,labels_v,obj)
    loss3 = cos_loss(d3,labels_v,obj)
    loss4 = cos_loss(d4,labels_v,obj)
    loss5 = cos_loss(d5,labels_v,obj)
    loss6 = cos_loss(d6,labels_v,obj)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	#print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    lossb0 = bce_loss(torch.sign(relu(d0-0.5)),labels_v)
    lossb1 = bce_loss(torch.sign(relu(d1-0.5)),labels_v)
    lossb2 = bce_loss(torch.sign(relu(d2-0.5)),labels_v)
    lossb3 = bce_loss(torch.sign(relu(d3-0.5)),labels_v)
    lossb4 = bce_loss(torch.sign(relu(d4-0.5)),labels_v)
    lossb5 = bce_loss(torch.sign(relu(d5-0.5)),labels_v)
    lossb6 = bce_loss(torch.sign(relu(d6-0.5)),labels_v)

    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    #print(torch.sign(relu(d0-0.5)))


    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    lossb = lossb0 + lossb1 + lossb2 + lossb3 + lossb4 + lossb5 + lossb6
    #print(loss,lossb)
    return loss0+lossb0,loss+lossb

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
    data_loader = get_loader('doc3djoint_nm_full')
    data_path = args.data_path
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols),augmentations=True)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=12, shuffle=True,pin_memory=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=12,pin_memory=True)

    # Setup Model

    model_boundary = get_model(args.arch_boundary, n_classes=1,in_channels=3) #Get the U-Net architecture
    model_boundary = torch.nn.DataParallel(model_boundary, device_ids=range(torch.cuda.device_count()))
    model_boundary.to(device)

    model_nm = get_model(args.arch_nm, n_classes=1,in_channels=3, img_size=(args.img_rows,args.img_cols)) #Get U2Net
    model_nm = torch.nn.DataParallel(model_nm, device_ids=[0,1,2])
    model_nm.to(device)
    
    model_bm = get_model(args.arch_bm, n_classes=2,in_channels=4, img_size=(args.bm_img_rows,args.bm_img_cols)) #Get mobileViT-rgbd
    model_bm = torch.nn.DataParallel(model_bm, device_ids=[0,1,2])
    model_bm.to(device)
    # Activation
    
    # optimizer_boundary
    #optimizer_nm= torch.optim.Adam(model_nm.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)
    #optimizer_boundary = torch.optim.Adam(model.parameters(), lr=args.l_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # LR Scheduler, which can reduce the learning rate as time passing by
    #sched_nm=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nm, mode='min', factor=0.5, patience=5, verbose=True)

    # Losses
    MSE = nn.MSELoss() #L2 Loss for measure the training loss and validation loss
    #loss_fn = nn.DepthLoss() # Depth Loss used throughout the whole training process, including L_C, L_D, L_T (Both in the first and second training phase)
    #gloss= grad_loss.Gradloss(window_size=5,padding=2) #The Gradient Loss used in L_C, i.e. deltaChead-deltaC

    epoch_start=0
    if args.resume_boundary is not None:                               
        if os.path.isfile(args.resume_boundary):# Loading model and optimizer_boundary from the checkpoint
            print("Loading model and optimizer_nm from checkpoint '{}'".format(args.resume_boundary))
            checkpoint = torch.load(args.resume_boundary)
            (model_boundary).load_state_dict(checkpoint['model_state'])
            #optimizer_boundary.load_state_dict(checkpoint['optimizer_boundary_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                .format(args.resume_boundary, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume_boundary)) 
    if args.resume_nm is not None:                               
        if os.path.isfile(args.resume_nm):# Loading model and optimizer_boundary from the checkpoint
            print("Loading model and optimizer_nm from checkpoint '{}'".format(args.resume_nm))
            checkpoint = torch.load(args.resume_nm)
            (model_nm).load_state_dict(checkpoint['model_state'])
            #optimizer_boundary.load_state_dict(checkpoint['optimizer_boundary_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                .format(args.resume_nm, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume_nm)) 
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
    model_joint=get_model(name='nm_u2net_joint_mobilevit_full',boundary_model=model_boundary, nm_model=model_nm,bm_model=model_bm)
    optimizer_joint= torch.optim.Adam(model_joint.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)
    sched_joint=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_joint, mode='min', factor=0.5, patience=5, verbose=True)
    optimizer_joint.load_state_dict(checkpoint['optimizer_state'])
    
    #Log file:
    if not os.path.exists(args.logdir_boundary):
        os.makedirs(args.logdir_boundary)
    experiment_name='joint_nm'
    log_file_name_boundary=os.path.join(args.logdir_boundary,experiment_name+'.txt')
    if os.path.isfile(log_file_name_boundary):
        log_file=open(log_file_name_boundary,'a')
    else:
        log_file=open(log_file_name_boundary,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    if not os.path.exists(args.logdir_nm):
        os.makedirs(args.logdir_nm)
    experiment_name='joint_nm'
    log_file_name_nm=os.path.join(args.logdir_nm,experiment_name+'.txt')
    if os.path.isfile(log_file_name_nm):
        log_file=open(log_file_name_nm,'a')
    else:
        log_file=open(log_file_name_nm,'w+')

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

    
    best_val_loss_boundary = 99999.0
    best_val_loss_nm = 99999.0
    best_val_loss_bm = 99999.0
    global_step=0

    print(get_lr(optimizer_joint))


    for epoch in range(epoch_start,args.n_epoch):
        avg_loss_boundary=0.0
        avg_loss2_boundary=0.0
        avg_lossall_boundary=0.0

        avg_loss_nm=0.0
        avg_loss2_nm=0.0
        avg_lossall_nm=0.0

        avg_loss_bm=0.0
        avgl1loss_bm=0.0


        for i, (images, norm, bm) in enumerate(trainloader):
            model_joint.train()
            images = Variable(images.to(device))

            #-------------------DEPTH--------------------------
            norm = Variable(norm.to(device))
            optimizer_joint.zero_grad()

            labels_msk=1-((norm[:,0,:,:]==0)&(norm[:,1,:,:]==0)&(norm[:,2,:,:]==0)).to(torch.float)
            msk_index=labels_msk.nonzero()
            #print(msk_index.shape)

            b1,b2,b3,b4,b5,b6,b7, d0, d1, d2, d3, d4, d5, d6, pred_bm = model_joint(images,(args.bm_img_rows,args.bm_img_cols))
            loss2_boundary, loss_boundary = muti_bce_loss_fusion(b1,b2,b3,b4,b5,b6,b7, labels_msk.unsqueeze(1))
            avg_loss2_boundary+=float(loss2_boundary)
            avg_lossall_boundary+=float(loss_boundary)
            avg_loss_boundary+=float(loss_boundary)

            pred_msk=(b1>0.5).to(torch.float32)
            #pred_msk_index=pred_msk.nonzero()

            only_d0=d0[msk_index[:,0],:,msk_index[:,1],msk_index[:,2]]
            only_d1=d1[msk_index[:,0],:,msk_index[:,1],msk_index[:,2]]
            only_d2=d2[msk_index[:,0],:,msk_index[:,1],msk_index[:,2]]
            only_d3=d3[msk_index[:,0],:,msk_index[:,1],msk_index[:,2]]
            only_d4=d4[msk_index[:,0],:,msk_index[:,1],msk_index[:,2]]
            only_d5=d5[msk_index[:,0],:,msk_index[:,1],msk_index[:,2]]
            only_d6=d6[msk_index[:,0],:,msk_index[:,1],msk_index[:,2]]
            only_d_lbl=norm[msk_index[:,0],:,msk_index[:,1],msk_index[:,2]]
            #print(only_d0.shape)
            #print(only_d0)

            loss2_nm, loss_nm = muti_cos_loss_fusion(only_d0, only_d1, only_d2, only_d3, only_d4, only_d5, only_d6, only_d_lbl)

            avg_loss2_nm+=float(loss2_nm)
            avg_lossall_nm+=float(loss_nm)
            avg_loss_nm+=float(loss_nm)

            #-------------------BM--------------------------
            bm = Variable(bm.to(device))
            predbm_nhwc = pred_bm.permute(0,2,3,1)
            loss_bm = l1_loss(predbm_nhwc, bm)

            avgl1loss_bm+=float(loss_bm)        
            avg_loss_bm+=float(loss_bm)

            loss=loss_boundary+loss_nm+loss_bm


            loss.backward()
            optimizer_joint.step()
            global_step+=1

            if (i+1) % 50 == 0:
                print("Epoch[%d/%d] Batch [%d/%d] Boundary Loss: %.4f Depth Loss: %.4f BM Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader),avg_loss_boundary/50.0, avg_loss_nm/50.0, avg_loss_bm/50.0 ))
                avg_loss_nm=0.0
                avg_loss_bm=0.0
                avg_loss_boundary=0.0

            if args.tboard and  (i+1) % 20 == 0:
                #show_wc_tnsboard(global_step, writer,images,labels,pred, 8,'Train Inputs', 'Train Depths', 'Train Pred. Depths')
                writer.add_scalar('Boundary: BCE Loss/train', avg_lossall_boundary/(i+1), global_step)
                writer.add_scalar('Depth: L1 Loss/train', avg_lossall_nm/(i+1), global_step)
                writer.add_scalar('BM: L1 Loss/train', avgl1loss_bm/(i+1), global_step)
            del b1,b2,b3,b4,b5,b6,b7,d0, d1, d2, d3, d4, d5, d6,only_d0, only_d1, only_d2, only_d3, only_d4, only_d5, only_d6,only_d_lbl, pred_bm,predbm_nhwc,pred_msk,loss,loss_bm,loss_boundary,loss_nm,loss2_boundary,loss2_nm
            if (i+1) % 50==0:
                torch.cuda.empty_cache()

        avg_loss2_boundary=avg_loss2_boundary/len(trainloader)
        avg_loss2_nm=avg_loss2_nm/len(trainloader)
        avgl1loss_bm=avgl1loss_bm/len(trainloader)
        #avg_gloss=avg_gloss/len(trainloader)
        print("Training Boundary loss2:%4f nm loss2:%4f BM L1:%4f" %(avg_loss2_boundary, avg_loss2_nm, avgl1loss_bm))

        train_losses_boundary=[avg_loss2_boundary]
        lrate_boundary=get_lr(optimizer_joint)
        write_log_file(log_file_name_boundary, train_losses_boundary, epoch+1, lrate_boundary,'Train')

        train_losses_nm=[avg_loss2_nm]
        lrate_nm=get_lr(optimizer_joint)
        write_log_file(log_file_name_nm, train_losses_nm, epoch+1, lrate_nm,'Train')
        
        train_losses_bm=[avgl1loss_bm]
        lrate_bm=get_lr(optimizer_joint)
        write_log_file(log_file_name_bm, train_losses_bm,epoch+1, lrate_bm,'Train')
        

        #-----------------EVALUATION-----------------
        model_joint.eval()
        
        val_loss_boundary=0.0
        val_loss_nm=0.0
        val_loss_bm=0.0
        for i_val, (images_val, nm_val, bm_val) in tqdm(enumerate(valloader)): #use progress bar
            with torch.no_grad():
                images_val = Variable(images_val.to(device))
                #-------------------DEPTH--------------------------
                nm_val = Variable(nm_val.to(device))

                labels_msk=1-((nm_val[:,0,:,:]==0)&(nm_val[:,1,:,:]==0)&(nm_val[:,2,:,:]==0)).to(torch.float)
                msk_index=labels_msk.nonzero()

                b1,b2,b3,b4,b5,b6,b7,d1,d2,d3,d4,d5,d6,d7,pred_bm= model_joint(images_val,(args.bm_img_rows,args.bm_img_cols))
                pred_msk=(b1>0.5).to(torch.float32)

                pred_msk=pred_msk.cpu()
                labels_msk=labels_msk.unsqueeze(1).cpu()

                loss_boundary = bce_loss(pred_msk, labels_msk)
                val_loss_boundary+=float(loss_boundary)

                d1=d1.cpu()
                nm_val=nm_val.cpu()
                only_d1=d1[msk_index[:,0],:,msk_index[:,1],msk_index[:,2]]
                only_d_lbl=nm_val[msk_index[:,0],:,msk_index[:,1],msk_index[:,2]]

                loss_nm=l1_loss(only_d1,only_d_lbl)
                val_loss_nm+=float(loss_nm)

                #-------------------BM------------------------
                predbm_nhwc = pred_bm.permute(0,2,3,1)
                l1loss = l1_loss(predbm_nhwc.cpu(), bm_val.cpu())
                #rloss,ssim,uworg,uwpred = reconst_loss(images_val[:,:-1,:,:],target_nhwc,labels_val)
                val_loss_bm+=float(l1loss)              
                
                del b1,b2,b3,b4,b5,b6,b7,d1,d2,d3,d4,d5,d6,d7,only_d_lbl,only_d1,labels_msk,predbm_nhwc,pred_bm,pred_msk
                if (i_val+1) % 50==0:
                    torch.cuda.empty_cache()

        if args.tboard:
            #show_wc_tnsboard(epoch+1, writer,images_val,labels_val,pred, 8,'Val Inputs', 'Val Depths', 'Val Pred. Depths')
            writer.add_scalar('Boundary: BCE Loss/val', val_loss_boundary/len(valloader), epoch+1)
            writer.add_scalar('Depth: L1 Loss/val', val_loss_nm/len(valloader), epoch+1)
            writer.add_scalar('BM: L1 Loss/val', val_loss_bm/len(valloader), epoch+1)

        val_loss_boundary=val_loss_boundary/len(valloader)
        val_loss_nm=val_loss_nm/len(valloader)
        val_loss_bm=val_loss_bm/len(valloader)
        print("val loss at epoch {}:: {}".format(epoch+1,val_loss_boundary))
        print("Depth val loss at epoch {}:: {}".format(epoch+1,val_loss_nm))
        print("BM val loss at epoch {}:: {}".format(epoch+1,val_loss_bm))

        val_losses_boundary=[val_loss_boundary]
        write_log_file(log_file_name_boundary, val_losses_boundary, epoch+1, lrate_boundary, 'Val')

        val_losses_nm=[val_loss_nm]
        write_log_file(log_file_name_nm, val_losses_nm, epoch+1, lrate_nm, 'Val')

        val_losses_bm=[val_loss_bm]
        write_log_file(log_file_name_bm, val_losses_bm, epoch+1, lrate_bm, 'Val')

        #reduce learning rate
        sched_joint.step(val_loss_bm)


        if val_loss_boundary<best_val_loss_boundary:
            best_val_loss_boundary=val_loss_boundary

        if val_loss_nm<best_val_loss_nm:
            best_val_loss_nm=val_loss_nm
        
        if val_loss_bm < best_val_loss_bm:
            best_val_loss_bm=val_loss_bm
        state = {'epoch': epoch+1,
                    'model_state': (model_joint.boundary_model).state_dict(),
                    'optimizer_state' : optimizer_joint.state_dict(),}
        torch.save(state, args.logdir_boundary+"{}_{}_{}_{}_best_model.pkl".format(args.arch_boundary, epoch+1,val_loss_boundary, experiment_name))
        state = {'epoch': epoch+1,
                    'model_state': (model_joint.nm_model).state_dict(),
                    'optimizer_state' : optimizer_joint.state_dict(),}
        torch.save(state, args.logdir_nm+"{}_{}_{}_{}_best_model.pkl".format(args.arch_nm, epoch+1,val_loss_nm, experiment_name))
        state = {'epoch': epoch+1,
                    'model_state': (model_joint.bm_model).state_dict(),
                    'optimizer_state' : optimizer_joint.state_dict(),}
        torch.save(state, args.logdir_bm+"{}_{}_{}_{}_best_model.pkl".format(args.arch_bm, epoch+1,val_loss_bm,experiment_name))
        torch.cuda.empty_cache()
        time.sleep(0.003)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch_boundary', nargs='?', type=str, default='u2net', 
                        help='Architecture to use')
    parser.add_argument('--arch_nm', nargs='?', type=str, default='u2net_nm', 
                        help='Architecture to use')
    parser.add_argument('--arch_bm', nargs='?', type=str, default='mobilevit_sandfcn_skip', 
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
    parser.add_argument('--batch_size', nargs='?', type=int, default=32, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-06, 
                        help='Learning Rate')
    parser.add_argument('--resume_nm', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--resume_bm', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--resume_boundary', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir_nm', nargs='?', type=str, default='./checkpoints-joint-nm_nm/',    
                        help='Path to store the loss logs')
    parser.add_argument('--logdir_bm', nargs='?', type=str, default='./checkpoints-joint-nm_bm/',    
                        help='Path to store the loss logs')
    parser.add_argument('--logdir_boundary', nargs='?', type=str, default='./checkpoints-joint-nm_boundary/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.set_defaults(tboard=True)

    args = parser.parse_args()
    train(args)


#CUDA_VISIBLE_DEVICES=0,1,2 python trainjoint_nm_full.py --data_path /disk2/sinan/doc3d/ --tboard --resume_boundary /home/sinan/DewarpNet-master/checkpoints-joint-nm_boundary/u2net_136_0.04724174547500123_joint_nm_best_model.pkl --resume_nm /home/sinan/DewarpNet-master/checkpoints-joint-nm_nm/u2net_nm_136_0.3566189971023474_joint_nm_best_model.pkl --resume_bm /home/sinan/DewarpNet-master/checkpoints-joint-nm_bm/mobilevit_sandfcn_skip_136_0.013469635993956376_joint_nm_best_model.pkl