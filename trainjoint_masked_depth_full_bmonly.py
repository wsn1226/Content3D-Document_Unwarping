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
from utils import show_wc_tnsboard,  get_lr, convert_state_dict

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
    del loss1, loss2, loss3, loss4, loss5, loss6
    return loss0, loss


def write_log_file(log_file_name,losses, epoch, lrate, phase):
    with open(log_file_name,'a') as f:
        f.write("\n{} LRate: {} Epoch: {} Loss: {}".format(phase, lrate, epoch, losses[0]))


def train(args):

    # Setup Dataloader
    data_loader = get_loader('doc3djoint_masked_depth_full')
    data_path = args.data_path
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols),augmentations=True)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=12, shuffle=True,pin_memory=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=12,pin_memory=True)

    # Setup Model

    model_boundary = get_model(args.arch_boundary, n_classes=1,in_channels=3) #Get the U-Net architecture
    model_boundary = torch.nn.DataParallel(model_boundary, device_ids=range(torch.cuda.device_count()))
    #boundary_state = convert_state_dict(torch.load(args.resume_boundary)['model_state'])
    #model_boundary.load_state_dict(boundary_state)
    model_boundary.to(device)

    model_depth = get_model(args.arch_depth, n_classes=1,in_channels=3, img_size=(args.img_rows,args.img_cols)) #Get U2Net
    model_depth = torch.nn.DataParallel(model_depth, device_ids=range(torch.cuda.device_count()))
    model_depth.to(device)
    
    model_bm = get_model(args.arch_bm, n_classes=2,in_channels=4, img_size=(args.bm_img_rows,args.bm_img_cols)) #Get mobileViT-rgbd
    model_bm = torch.nn.DataParallel(model_bm, device_ids=range(torch.cuda.device_count()))
    model_bm.to(device)
    # Activation
    
    # optimizer_boundary
    optimizer= torch.optim.Adam(model_bm.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)
    #optimizer_depth= torch.optim.Adam(model_depth.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)
    #optimizer_boundary = torch.optim.Adam(model.parameters(), lr=args.l_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # LR Scheduler, which can reduce the learning rate as time passing by
    sched_bm=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Losses
    #MSE = nn.MSELoss() #L2 Loss for measure the training loss and validation loss
    #loss_fn = nn.DepthLoss() # Depth Loss used throughout the whole training process, including L_C, L_D, L_T (Both in the first and second training phase)
    #gloss= grad_loss.Gradloss(window_size=5,padding=2) #The Gradient Loss used in L_C, i.e. deltaChead-deltaC

    epoch_start=0
    if args.resume_boundary is not None:                               
        if os.path.isfile(args.resume_boundary):# Loading model and optimizer_boundary from the checkpoint
            print("Loading model and optimizer_depth from checkpoint '{}'".format(args.resume_boundary))
            checkpoint = torch.load(args.resume_boundary)
            (model_boundary).load_state_dict(checkpoint['model_state'])
            #optimizer_boundary.load_state_dict(checkpoint['optimizer_boundary_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                .format(args.resume_boundary, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume_boundary))
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
            #optimizer.load_state_dict(checkpoint['optimizer_boundary_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                .format(args.resume_bm, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume_bm))
    #model_bm=get_model(name='u2net_joint_mobilevit_onlydbm_rgbdiff',depth_model=model_depth,bm_model=model_bm)
    #sched_joint=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    #optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    #Log file:
    experiment_name='joint_masked_depth'
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

    best_val_loss_bm = 99999.0
    global_step=0

    print(get_lr(optimizer))

    model_boundary.eval()
    model_depth.eval()

    for epoch in range(epoch_start,args.n_epoch):

        avg_loss_bm=0.0
        avgl1loss_bm=0.0

        for i, (images, dmap, bm) in enumerate(trainloader):
            model_bm.train()
            images = Variable(images.to(device))

            #-------------------DEPTH--------------------------
            #dmap = Variable(dmap.to(device))
            optimizer.zero_grad()

            #labels_msk=(dmap>=0).to(torch.float)
            #print(msk_index.shape)
            pred_msk=(model_boundary(images[:,[2,1,0],:,:])[0]>0.5).to(torch.float)
            #d0, d1, d2, d3, d4, d5, d6, pred_bm = model_bm(pred_msk, images,(args.bm_img_rows,args.bm_img_cols))

            extracted_img=torch.mul(pred_msk.repeat(1,3,1,1),images)
            alb0 = extracted_img[:,:,1:,:]-extracted_img[:,:,:-1,:]
            alb1 = extracted_img[:,:,:,1:]-extracted_img[:,:,:,:-1]
            alb0 = F.pad(alb0, ((0,0,1,0,0,0,0,0)), mode='constant')
            alb1 = F.pad(alb1, ((1,0,0,0,0,0,0,0)), mode='constant')
            alb0 = torch.abs(F.interpolate(alb0, (args.bm_img_rows,args.bm_img_cols), mode='bilinear', align_corners=False))
            alb1 = torch.abs(F.interpolate(alb1, (args.bm_img_rows,args.bm_img_cols), mode='bilinear', align_corners=False))
            alb = (alb0+alb1).to(torch.float)
            maxalb=torch.max(alb)
            alb/=maxalb
            #cv2.imwrite('testalb.png',alb[0].permute(1,2,0).cpu().numpy()*255)
            d1=relu(model_depth(extracted_img)[0])
            masked_depth=pred_msk*d1
            bm_inputdepth=F.interpolate(masked_depth,(args.bm_img_rows,args.bm_img_cols),mode='nearest')
            alb=alb.detach()
            bm_inputdepth=bm_inputdepth.detach()
            rgbd=torch.cat([alb,bm_inputdepth],dim=1)
            del masked_depth, extracted_img, bm_inputdepth, alb, alb0, alb1,dmap
            rgbd=Variable(rgbd)
            #print(rgbd)
            #bm_input=F.interpolate(rgbd,bm_img_size,mode='nearest')
            pred_bm=model_bm(rgbd)

            #-------------------BM--------------------------
            bm = Variable(bm.to(device))
            predbm_nhwc = pred_bm.permute(0,2,3,1)
            loss_bm = l1_loss(predbm_nhwc, bm)
            #print(loss_bm)

            avgl1loss_bm+=float(loss_bm)        
            avg_loss_bm+=float(loss_bm)

            loss=loss_bm

            loss.backward()
            optimizer.step()
            global_step+=1

            if (i+1) % 50 == 0:
                print("Epoch[%d/%d] Batch [%d/%d] BM Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader),avg_loss_bm/50.0 ))
                #avg_loss_depth=0.0
                avg_loss_bm=0.0
                #avg_loss_boundary=0.0

            if args.tboard and  (i+1) % 20 == 0:
                #show_wc_tnsboard(global_step, writer,images,labels,pred, 8,'Train Inputs', 'Train Depths', 'Train Pred. Depths')
                #writer.add_scalar('Boundary: BCE Loss/train', avg_lossall_boundary/(i+1), global_step)
                #writer.add_scalar('Depth: L1 Loss/train', avg_lossall_depth/(i+1), global_step)
                writer.add_scalar('BM: L1 Loss/train', avgl1loss_bm/(i+1), global_step)
            del d1,rgbd,images, bm, pred_bm,predbm_nhwc,pred_msk,loss,loss_bm#,loss_depth,loss2_depth
            if (i+1)%50==0:
                torch.cuda.empty_cache()
        #avg_loss2_boundary=avg_loss2_boundary/len(trainloader)
        #avg_loss2_depth=avg_loss2_depth/len(trainloader)
        avgl1loss_bm=avgl1loss_bm/len(trainloader)
        #avg_gloss=avg_gloss/len(trainloader)
        print("Training Loss L1:%4f" %(avgl1loss_bm))
        
        train_losses_bm=[avgl1loss_bm]
        lrate_bm=get_lr(optimizer)
        write_log_file(log_file_name_bm, train_losses_bm,epoch+1, lrate_bm,'Train')
        

        #-----------------EVALUATION-----------------
        model_bm.eval()
        
        #val_loss_boundary=0.0
        #val_loss_depth=0.0
        val_loss_bm=0.0
        for i_val, (images_val, depth_val, bm_val) in tqdm(enumerate(valloader)): #use progress bar
            with torch.no_grad():
                images_val = images_val.to(device)
                pred_msk=(model_boundary(images_val[:,[2,1,0],:,:])[0]>0.5).to(torch.float)

                extracted_img=torch.mul(pred_msk.repeat(1,3,1,1),images_val)
                alb0 = extracted_img[:,:,1:,:]-extracted_img[:,:,:-1,:]
                alb1 = extracted_img[:,:,:,1:]-extracted_img[:,:,:,:-1]
                alb0 = F.pad(alb0, ((0,0,1,0,0,0,0,0)), mode='constant')
                alb1 = F.pad(alb1, ((1,0,0,0,0,0,0,0)), mode='constant')
                alb0 = torch.abs(F.interpolate(alb0, (args.bm_img_rows,args.bm_img_cols), mode='bilinear', align_corners=False))
                alb1 = torch.abs(F.interpolate(alb1, (args.bm_img_rows,args.bm_img_cols), mode='bilinear', align_corners=False))
                alb = (alb0+alb1).to(torch.float)
                maxalb=torch.max(alb)
                alb/=maxalb
                #cv2.imwrite('testalb.png',alb[0].permute(1,2,0).cpu().numpy()*255)
                d1=relu(model_depth(extracted_img)[0])
                masked_depth=pred_msk*d1
                bm_inputdepth=F.interpolate(masked_depth,(args.bm_img_rows,args.bm_img_cols),mode='nearest')
                rgbd=torch.cat([alb,bm_inputdepth],dim=1)
                #bm_input=F.interpolate(rgbd,bm_img_size,mode='nearest')
                pred_bm=model_bm(rgbd)

                #-------------------BM------------------------
                predbm_nhwc = pred_bm.permute(0,2,3,1).cpu()
                bm_val=bm_val.cpu()
                l1loss = l1_loss(predbm_nhwc, bm_val)
                #rloss,ssim,uworg,uwpred = reconst_loss(images_val[:,:-1,:,:],target_nhwc,labels_val)
                val_loss_bm+=float(l1loss)              
                
                del images_val, depth_val, bm_val,predbm_nhwc,pred_bm,pred_msk,masked_depth, rgbd, extracted_img, bm_inputdepth, alb, alb0, alb1
                if (i_val+1) % 50==0:
                    torch.cuda.empty_cache()

        if args.tboard:
            #show_wc_tnsboard(epoch+1, writer,images_val,labels_val,pred, 8,'Val Inputs', 'Val Depths', 'Val Pred. Depths')
            #writer.add_scalar('Boundary: BCE Loss/val', val_loss_boundary/len(valloader), epoch+1)
            #writer.add_scalar('Depth: L1 Loss/val', val_loss_depth/len(valloader), epoch+1)
            writer.add_scalar('BM: L1 Loss/val', val_loss_bm/len(valloader), epoch+1)

        #val_loss_boundary=val_loss_boundary/len(valloader)
        #val_loss_depth=val_loss_depth/len(valloader)
        val_loss_bm=val_loss_bm/len(valloader)
        #print("val loss at epoch {}:: {}".format(epoch+1,val_loss_boundary))
        #print("Depth val loss at epoch {}:: {}".format(epoch+1,val_loss_depth))
        print("BM val loss at epoch {}:: {}".format(epoch+1,val_loss_bm))

        #val_losses_boundary=[val_loss_boundary]
        #write_log_file(log_file_name_boundary, val_losses_boundary, epoch+1, lrate_boundary, 'Val')

        val_losses_bm=[val_loss_bm]
        write_log_file(log_file_name_bm, val_losses_bm, epoch+1, lrate_bm, 'Val')

        #reduce learning rate
        sched_bm.step(val_loss_bm)


        #if val_loss_boundary<best_val_loss_boundary:
            #best_val_loss_boundary=val_loss_boundary
        
        if val_loss_bm < best_val_loss_bm:
            best_val_loss_bm=val_loss_bm
        state = {'epoch': epoch+1,
                    'model_state': (model_bm.bm_model).state_dict(),
                    'optimizer_state' : optimizer.state_dict(),}
        torch.save(state, args.logdir_bm+"{}_{}_{}_{}_best_model.pkl".format(args.arch_bm, epoch+1,val_loss_bm,experiment_name))
        torch.cuda.empty_cache()
        time.sleep(0.003)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch_boundary', nargs='?', type=str, default='u2net', 
                        help='Architecture to use')
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
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-06, 
                        help='Learning Rate')
    parser.add_argument('--resume_depth', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--resume_bm', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--resume_boundary', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir_depth', nargs='?', type=str, default='./checkpoints-jointmskdepth_rgbdiff-u2depth_full_onlytrainbm/',    
                        help='Path to store the loss logs')
    parser.add_argument('--logdir_bm', nargs='?', type=str, default='./checkpoints-jointmskdepth_rgbdiff-bm_full_onlytrainbm/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.set_defaults(tboard=True)

    args = parser.parse_args()
    train(args)


#CUDA_VISIBLE_DEVICES=1,2,0 python trainjoint_masked_depth_full_bmonly.py --data_path /disk2/sinan/doc3d/ --batch_size 38 --tboard --resume_boundary /home/sinan/DewarpNet-master/checkpoints-u2net-nopre/u2net_132_0.010844790506031037_0.04020660579093357_boundary_best_model.pkl --resume_depth /home/sinan/DewarpNet-master/checkpoints-jointmskdepth-u2depth_full/u2net_depth_239_0.03534626894363082_joint_masked_depth_best_model.pkl --resume_bm /home/sinan/DewarpNet-master/checkpoints-jointmskdepth_rgbdiff-bm_full/mobilevit_sandfcn_skipRGBD_132_0.012492866956709344_joint_masked_depth_best_model.pkl