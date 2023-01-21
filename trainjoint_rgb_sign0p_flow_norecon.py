# code to train world coord regression from RGB Image
# models are saved in checkpoints-wc/

import sys, os
from this import d
from cv2 import BFMatcher_create, boundingRect
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
import recon_lossc
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

def write_log_file(log_file_name,losses, epoch, lrate, phase):
    with open(log_file_name,'a') as f:
        f.write("\n{} LRate: {} Epoch: {} Loss: {}".format(phase, lrate, epoch, losses[0]))


def train(args):

    # Setup Dataloader
    data_loader = get_loader('originalimg_dmap_flow')
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

    model_prob = get_model(args.arch_prob, n_classes=1,in_channels=3, img_size=(args.img_rows,args.img_cols)) #Get U2Net
    model_prob = torch.nn.DataParallel(model_prob, device_ids=[0,1,2])
    model_prob.to(device)
    
    model_flow = get_model(args.arch_flow, n_classes=2,in_channels=4, img_size=(args.flow_img_rows,args.flow_img_cols)) #Get mobileViT-rgbd
    model_flow = torch.nn.DataParallel(model_flow, device_ids=[0,1,2])
    model_flow.to(device)
    # Activation
    
    # optimizer_boundary
    #optimizer_prob= torch.optim.Adam(model_prob.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)
    #optimizer_boundary = torch.optim.Adam(model.parameters(), lr=args.l_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # LR Scheduler, which can reduce the learning rate as time passing by
    #sched_prob=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_prob, mode='min', factor=0.5, patience=5, verbose=True)

    # Losses
    MSE = nn.MSELoss() #L2 Loss for measure the training loss and validation loss
    #loss_fn = nn.DepthLoss() # Depth Loss used throughout the whole training process, including L_C, L_D, L_T (Both in the first and second training phase)
    reconst_loss= recon_lossc.Unwarploss()

    epoch_start=0
    if args.resume_boundary is not None:                               
        if os.path.isfile(args.resume_boundary):# Loading model and optimizer_boundary from the checkpoint
            print("Loading model and optimizer_prob from checkpoint '{}'".format(args.resume_boundary))
            checkpoint = torch.load(args.resume_boundary)
            (model_boundary).load_state_dict(checkpoint['model_state'])
            #optimizer_boundary.load_state_dict(checkpoint['optimizer_boundary_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                .format(args.resume_boundary, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume_boundary)) 
    if args.resume_prob is not None:                               
        if os.path.isfile(args.resume_prob):# Loading model and optimizer_boundary from the checkpoint
            print("Loading model and optimizer_prob from checkpoint '{}'".format(args.resume_prob))
            checkpoint = torch.load(args.resume_prob)
            (model_prob).load_state_dict(checkpoint['model_state'])
            #optimizer_boundary.load_state_dict(checkpoint['optimizer_boundary_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                .format(args.resume_prob, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume_prob)) 
    if args.resume_flow is not None:                               
        if os.path.isfile(args.resume_flow):# Loading model and optimizer_boundary from the checkpoint
            print("Loading model and optimizer_flow from checkpoint '{}'".format(args.resume_flow))
            checkpoint = torch.load(args.resume_flow)
            (model_flow).load_state_dict(checkpoint['model_state'])
            #optimizer_joint.load_state_dict(checkpoint['optimizer_boundary_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                .format(args.resume_flow, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume_flow))
    model_joint=get_model(name='u2net_joint_rgbp_norecon',boundary_model=model_boundary, prob_model=model_prob,flow_model=model_flow)
    optimizer_joint= torch.optim.Adam(model_joint.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)
    sched_joint=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_joint, mode='min', factor=0.5, patience=4, verbose=True)
    #optimizer_joint.load_state_dict(checkpoint['optimizer_state'])
    
    #Log file:
    if not os.path.exists(args.logdir_boundary):
        os.makedirs(args.logdir_boundary)
    experiment_name='u2net_joint_rgbp_norecon'
    log_file_name_boundary=os.path.join(args.logdir_boundary,experiment_name+'.txt')
    if os.path.isfile(log_file_name_boundary):
        log_file=open(log_file_name_boundary,'a')
    else:
        log_file=open(log_file_name_boundary,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    if not os.path.exists(args.logdir_prob):
        os.makedirs(args.logdir_prob)
    log_file_name_prob=os.path.join(args.logdir_prob,experiment_name+'.txt')
    if os.path.isfile(log_file_name_prob):
        log_file=open(log_file_name_prob,'a')
    else:
        log_file=open(log_file_name_prob,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    if not os.path.exists(args.logdir_flow):
        os.makedirs(args.logdir_flow)
    log_file_name_flow=os.path.join(args.logdir_flow,experiment_name+'.txt')
    if os.path.isfile(log_file_name_flow):
        log_file=open(log_file_name_flow,'a')
    else:
        log_file=open(log_file_name_flow,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    # Setup tensorboard for visualization
    if args.tboard:
        # save logs in runs/<experiment_name> 
        writer = SummaryWriter(comment=experiment_name)

    
    best_val_loss_boundary = 99999.0
    best_val_loss_prob = 99999.0
    best_val_loss_flow = 99999.0
    global_step=0

    print(get_lr(optimizer_joint))


    for epoch in range(epoch_start,args.n_epoch):
        avg_loss_boundary=0.0
        avg_loss2_boundary=0.0
        avg_lossall_boundary=0.0

        avg_loss_prob=0.0
        avgl1loss_prob=0.0

        avg_loss_flow=0.0
        avgl1loss_flow=0.0

        #avgrloss=0.0
        #avgimgrloss=0.0

        #avg_loss_flowfromrecon=0.0
        #avgl1loss_flowfromrecon=0.0

        #avg_loss_ad=0.0

        model_boundary.train()
        model_prob.train()
        model_flow.train()

        for i, (images, msk, pmap, flow) in enumerate(trainloader):
            images = Variable(images.to(device))
            #recon=Variable(recon.to(device))
            pmap = Variable(pmap.to(device))
            labels_msk=Variable(msk.to(device))
            #-------------------BOUNDARY--------------------------
            optimizer_joint.zero_grad()

            #print(msk_index.shape)

            b1,b2,b3,b4,b5,b6,b7,pred_prob,pred_flow= model_joint(images,(args.flow_img_rows,args.flow_img_cols))
            loss2_boundary, loss_boundary = muti_bce_loss_fusion(b1,b2,b3,b4,b5,b6,b7, labels_msk)
            avg_loss2_boundary+=float(loss2_boundary)
            avg_lossall_boundary+=float(loss_boundary)
            avg_loss_boundary+=float(loss_boundary)

            #-------------------PROBABILITY--------------------------
            loss_prob=l1_loss(pred_prob,pmap)

            avgl1loss_prob+=float(loss_prob)
            avg_loss_prob+=float(loss_prob)

            #-------------------BM--------------------------
            flow = Variable(flow.to(device))
            flow = F.interpolate(flow.permute(0,3,1,2),(128,128),mode='bilinear',align_corners=True).permute(0,2,3,1)
            predflow_nhwc = pred_flow.permute(0,2,3,1)
            #predflow_nhwcfromrecon = pred_flowfromrecon.permute(0,2,3,1)
            loss_flow = l1_loss(predflow_nhwc, flow)
            #loss_flowfromrecon = l1_loss(predflow_nhwcfromrecon, flow)
            #rloss= reconst_loss(recon,predflow_nhwc,flow)
            #imgrloss = reconst_loss(images,predflow_nhwc,flow)
            #loss_ad=l1_loss(predflow_nhwcfromrecon,predflow_nhwc)

            avgl1loss_flow+=float(loss_flow)        
            avg_loss_flow+=float(loss_flow)

            #avgl1loss_flowfromrecon+=float(loss_flowfromrecon)        
            #avg_loss_flowfromrecon+=float(loss_flowfromrecon)            
            #avgrloss+=float(rloss)
            #avgimgrloss+=float(imgrloss)
            #avg_loss_ad+=float(loss_ad)

            loss=loss_boundary+loss_prob+loss_flow#+loss_ad #+0.05*rloss+0.05*imgrloss  + loss_flowfromrecon 


            loss.backward()
            optimizer_joint.step()
            global_step+=1

            if (i+1) % 50 == 0:
                #print("Epoch[%d/%d] Batch [%d/%d] Boundary Loss: %.4f Depth Loss: %.4f BM Loss: %.4f Recon Loss: %.4f Img Recon: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader),avg_loss_boundary/50.0, avg_loss_prob/50.0, avg_loss_flow/50.0,avgrloss/50.0 ,avgimgrloss/50.0))
                print("Epoch[%d/%d] Batch [%d/%d] Boundary Loss: %.4f Prob Loss: %.4f Flow Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader),avg_loss_boundary/50.0, avg_loss_prob/50.0, avg_loss_flow/50.0))
                avg_loss_prob=0.0
                avg_loss_flow=0.0
                avg_loss_boundary=0.0
                #avg_loss_flowfromrecon=0.0
                #avg_loss_ad=0.0
                #avgrloss=0.0
                #avgimgrloss=0.0

            if args.tboard and  (i+1) % 20 == 0:
                #show_wc_tnsboard(global_step, writer,images,labels,pred, 8,'Train Inputs', 'Train Depths', 'Train Pred. Depths')
                writer.add_scalar('Boundary: BCE Loss/train', avg_lossall_boundary/(i+1), global_step)
                writer.add_scalar('Probability: L1 Loss/train', avgl1loss_prob/(i+1), global_step)
                writer.add_scalar('Flow: L1 Loss/train', avgl1loss_flow/(i+1), global_step)
                #writer.add_scalar('BM From Recon: L1 Loss/train', avgl1loss_flowfromrecon/(i+1), global_step)
            del b1,b2,b3,b4,b5,b6,b7,pred_prob, pmap, pred_flow,predflow_nhwc,loss,loss_flow,loss_boundary,loss_prob,loss2_boundary
            if (i+1) % 50==0:
                torch.cuda.empty_cache()

        avg_loss2_boundary=avg_loss2_boundary/len(trainloader)
        avgl1loss_prob=avgl1loss_prob/len(trainloader)
        avgl1loss_flow=avgl1loss_flow/len(trainloader)
        #avgl1loss_flowfromrecon=avgl1loss_flowfromrecon/len(trainloader)
        #avg_gloss=avg_gloss/len(trainloader)
        print("Training Boundary loss2:%4f prob loss:%4f Flow L1:%4f" %(avg_loss2_boundary, avgl1loss_prob, avgl1loss_flow))

        train_losses_boundary=[avg_loss2_boundary]
        lrate_boundary=get_lr(optimizer_joint)
        write_log_file(log_file_name_boundary, train_losses_boundary, epoch+1, lrate_boundary,'Train')

        train_losses_prob=[avgl1loss_prob]
        lrate_prob=get_lr(optimizer_joint)
        write_log_file(log_file_name_prob, train_losses_prob, epoch+1, lrate_prob,'Train')
        
        train_losses_flow=[avgl1loss_flow]
        lrate_flow=get_lr(optimizer_joint)
        write_log_file(log_file_name_flow, train_losses_flow,epoch+1, lrate_flow,'Train')
        

        #-----------------EVALUATION-----------------
        model_joint.eval()
        
        val_loss_boundary=0.0
        val_loss_prob=0.0
        val_loss_flow=0.0
        #val_l1loss_fromrecon=0.0
        for i_val, (images_val, msk, prob_val, flow_val) in tqdm(enumerate(valloader)): #use progress bar
            with torch.no_grad():
                images_val = Variable(images_val.to(device))
                #recon = Variable(recon.to(device))
                prob_val = Variable(prob_val.to(device))
                labels_msk=Variable(msk.to(device))
                #-------------------BOUNDARY--------------------------

                b1,b2,b3,b4,b5,b6,b7,pred_prob,pred_flow= model_joint(images_val,(args.flow_img_rows,args.flow_img_cols))
                pred_msk=(b1>0.5).to(torch.float32)

                pred_msk=pred_msk.cpu()
                labels_msk=labels_msk.cpu()

                loss_boundary = bce_loss(pred_msk, labels_msk)
                val_loss_boundary+=float(loss_boundary)

                #-------------------PROBABILITY--------------------------
                pred_prob=pred_prob.cpu()
                prob_val=prob_val.cpu()

                loss_prob=l1_loss(pred_prob,prob_val)
                val_loss_prob+=float(loss_prob)

                #-------------------BM------------------------
                flow_val = F.interpolate(flow_val.permute(0,3,1,2),(128,128),mode='bilinear',align_corners=True).permute(0,2,3,1)
                predflow_nhwc = pred_flow.permute(0,2,3,1)
                #predflow_nhwcfromrecon = pred_flowfromrecon.permute(0,2,3,1)
                predflow_nhwc=predflow_nhwc.cpu()
                #predflow_nhwcfromrecon=predflow_nhwcfromrecon.cpu()
                flow_val=flow_val.cpu()
                l1loss = l1_loss(predflow_nhwc, flow_val)
                #l1loss_fromrecon = l1_loss(predflow_nhwcfromrecon, flow_val)
                #rloss,ssim,uworg,uwpred = reconst_loss(images_val[:,:-1,:,:],target_nhwc,labels_val)
                val_loss_flow+=float(l1loss)
                #val_l1loss_fromrecon+=float(l1loss_fromrecon)              
                
                del b1,b2,b3,b4,b5,b6,b7,pred_prob,prob_val,labels_msk,predflow_nhwc,pred_flow,pred_msk#,pred_flowfromrecon,recon
                if (i_val+1) % 50==0:
                    torch.cuda.empty_cache()

        if args.tboard:
            #show_wc_tnsboard(epoch+1, writer,images_val,labels_val,pred, 8,'Val Inputs', 'Val Depths', 'Val Pred. Depths')
            writer.add_scalar('Boundary: BCE Loss/val', val_loss_boundary/len(valloader), epoch+1)
            writer.add_scalar('Probability: L1 Loss/val', val_loss_prob/len(valloader), epoch+1)
            writer.add_scalar('Flow: L1 Loss/val', val_loss_flow/len(valloader), epoch+1)
            #writer.add_scalar('BM From Recon: L1 Loss/val', val_l1loss_fromrecon/len(valloader), epoch+1)

        val_loss_boundary=val_loss_boundary/len(valloader)
        val_loss_prob=val_loss_prob/len(valloader)
        val_loss_flow=val_loss_flow/len(valloader)
        #val_l1loss_fromrecon=val_l1loss_fromrecon/len(valloader)
        print("val loss at epoch {}:: {}".format(epoch+1,val_loss_boundary))
        print("Probability val loss at epoch {}:: {}".format(epoch+1,val_loss_prob))
        print("Flow val loss at epoch {}:: {}".format(epoch+1,val_loss_flow))
        #print("val loss from recon at epoch {}:: {}".format(epoch+1,val_l1loss_fromrecon))

        val_losses_boundary=[val_loss_boundary]
        write_log_file(log_file_name_boundary, val_losses_boundary, epoch+1, lrate_boundary, 'Val')

        val_losses_prob=[val_loss_prob]
        write_log_file(log_file_name_prob, val_losses_prob, epoch+1, lrate_prob, 'Val')

        val_losses_flow=[val_loss_flow]
        write_log_file(log_file_name_flow, val_losses_flow, epoch+1, lrate_flow, 'Val')

        #reduce learning rate
        sched_joint.step(val_loss_flow)


        if val_loss_boundary<best_val_loss_boundary:
            best_val_loss_boundary=val_loss_boundary

        if val_loss_prob<best_val_loss_prob:
            best_val_loss_prob=val_loss_prob
        
        if val_loss_flow < best_val_loss_flow:
            best_val_loss_flow=val_loss_flow
        state = {'epoch': epoch+1,
                    'model_state': (model_joint.boundary_model).state_dict(),
                    'optimizer_state' : optimizer_joint.state_dict(),}
        torch.save(state, args.logdir_boundary+"{}_{}_{}_{}_best_model.pkl".format(args.arch_boundary, epoch+1,val_loss_boundary, experiment_name))
        state = {'epoch': epoch+1,
                    'model_state': (model_joint.prob_model).state_dict(),
                    'optimizer_state' : optimizer_joint.state_dict(),}
        torch.save(state, args.logdir_prob+"{}_{}_{}_{}_best_model.pkl".format(args.arch_prob, epoch+1,val_loss_prob, experiment_name))
        state = {'epoch': epoch+1,
                    'model_state': (model_joint.flow_model).state_dict(),
                    'optimizer_state' : optimizer_joint.state_dict(),}
        torch.save(state, args.logdir_flow+"{}_{}_{}_{}_best_model.pkl".format(args.arch_flow, epoch+1,val_loss_flow,experiment_name))
        torch.cuda.empty_cache()
        time.sleep(0.003)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch_boundary', nargs='?', type=str, default='u2net', 
                        help='Architecture to use')
    parser.add_argument('--arch_prob', nargs='?', type=str, default='mobilevit_sandfcn_skip_prob_sign', 
                        help='Architecture to use')
    parser.add_argument('--arch_flow', nargs='?', type=str, default='mobilevit_sandfcn_skipRGBD', 
                        help='Architecture to use')
    parser.add_argument('--data_path', nargs='?', type=str, default='', 
                        help='Data path to load data')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Width of the input image')
    parser.add_argument('--flow_img_rows', nargs='?', type=int, default=128, 
                        help='Height of the input image')
    parser.add_argument('--flow_img_cols', nargs='?', type=int, default=128, 
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=400, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=200, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--resume_prob', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--resume_flow', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--resume_boundary', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir_prob', nargs='?', type=str, default='./checkpoints-joint_rgb_sign0_prob/',    
                        help='Path to store the loss logs')
    parser.add_argument('--logdir_flow', nargs='?', type=str, default='./checkpoints-joint_rgb_sign0p_flow/',    
                        help='Path to store the loss logs')
    parser.add_argument('--logdir_boundary', nargs='?', type=str, default='./checkpoints-joint_rgb_sign0p_boundary/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.set_defaults(tboard=True)

    args = parser.parse_args()
    train(args)


#CUDA_VISIBLE_DEVICES=1,0,2 python trainjoint_rgb_sign0p_flow_norecon.py --data_path /disk2/sinan/doc3d/ --batch_size 40 --tboard --resume_boundary /home/sinan/DewarpNet-master/checkpoints-joint_rgb_sign0p_boundary/u2net_73_0.04100353798270225_u2net_joint_rgbp_norecon_best_model.pkl --resume_prob /home/sinan/DewarpNet-master/checkpoints-joint_rgb_sign0_prob/mobilevit_sandfcn_skip_prob_sign_73_0.041875813722610476_u2net_joint_rgbp_norecon_best_model.pkl --resume_flow /home/sinan/DewarpNet-master/checkpoints-joint_rgb_sign0p_flow/mobilevit_sandfcn_skipRGBD_73_0.01418123384192586_u2net_joint_rgbp_norecon_best_model.pkl