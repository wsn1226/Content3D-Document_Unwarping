# code to train world coord regression from RGB Image
# models are saved in checkpoints-wc/

import sys, os
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

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    #print(loss,lossb)
    return loss0, loss


def write_log_file(log_file_name,losses, epoch, lrate, phase):
    with open(log_file_name,'a') as f:
        f.write("\n{} LRate: {} Epoch: {} Loss: {}".format(phase, lrate, epoch, losses[0]))


def train(args):

    # Setup Dataloader
    data_loader = get_loader('doc3ddepth')
    data_path = args.data_path
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=True)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=4, shuffle=True,pin_memory=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=4,pin_memory=True)

    # Setup Model
    model = get_model(args.arch, n_classes,in_channels=3) #Get the U-Net architecture
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    # Activation
    
    # Optimizer
    #optimizer= torch.optim.Adam(model.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # LR Scheduler, which can reduce the learning rate as time passing by
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Losses
    MSE = nn.MSELoss() #L2 Loss for measure the training loss and validation loss
    #loss_fn = nn.DepthLoss() # Depth Loss used throughout the whole training process, including L_C, L_D, L_T (Both in the first and second training phase)
    #gloss= grad_loss.Gradloss(window_size=5,padding=2) #The Gradient Loss used in L_C, i.e. deltaChead-deltaC

    epoch_start=0
    if args.resume is not None:
        if args.resume[-4:]=='.pth':
            print("load from the initial pth file")
            checkpoint=torch.load(args.resume)
            #print(checkpoint)
            model.load_state_dict(checkpoint,False)
        else:                                     
            if os.path.isfile(args.resume):# Loading model and optimizer from the checkpoint
                print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['model_state'])
                #optimizer.load_state_dict(checkpoint['optimizer_state'])
                print("Loaded checkpoint '{}' (epoch {})"                    
                    .format(args.resume, checkpoint['epoch']))
                epoch_start=checkpoint['epoch']
            else:
                print("No checkpoint found at '{}'".format(args.resume)) 
    
    #Log file:
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    experiment_name='depth' #activation_dataset_lossparams_augmentations_trainstart
    log_file_name=os.path.join(args.logdir,experiment_name+'.txt')
    if os.path.isfile(log_file_name):
        log_file=open(log_file_name,'a')
    else:
        log_file=open(log_file_name,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    # Setup tensorboard for visualization
    if args.tboard:
        # save logs in runs/<experiment_name> 
        writer = SummaryWriter(comment=experiment_name)

    best_val_loss = 99999.0
    global_step=0

    print("Learning rate: ",get_lr(optimizer))

    for epoch in range(epoch_start,args.n_epoch):
        avg_loss=0.0
        avg_loss2=0.0
        avg_lossall=0.0

        model.train() # Tell the model we're training

        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            #print(torch.max(labels))

            optimizer.zero_grad()
            labels_msk=torch.sign(relu(labels))
            #print(labels_msk)
            msk_index=(labels_msk).nonzero()
            #print(msk_index.shape)

            d0, d1, d2, d3, d4, d5, d6 = model(images)
            #print(d0)
            #loss2_general, loss_general = muti_l1_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
            d0_msk, d1_msk, d2_msk, d3_msk, d4_msk, d5_msk, d6_msk=(torch.sign(relu(d0))) , (torch.sign(relu(d1))) , (torch.sign(relu(d2))) , (torch.sign(relu(d3))) , (torch.sign(relu(d4))) , (torch.sign(relu(d5))) , (torch.sign(relu(d6)))
            #print(d0_msk)
            #print(d0[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]].shape)
            loss2_msk, loss_msk = muti_bce_loss_fusion(d0_msk, d1_msk, d2_msk, d3_msk, d4_msk, d5_msk, d6_msk, labels_msk)

            only_d0=d0[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d1=d1[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d2=d2[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d3=d3[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d4=d4[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d5=d5[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d6=d6[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            only_d_lbl=labels[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
            #print(only_d0)

            loss2_depth, loss_depth = muti_l1_loss_fusion(only_d0, only_d1, only_d2, only_d3, only_d4, only_d5, only_d6, only_d_lbl)

            #print(loss_msk)
            loss=loss_depth
            avg_loss2+=float(loss2_depth+loss2_msk)
            avg_lossall+=float(loss)
            avg_loss+=float(loss)

            loss.backward()
            optimizer.step()
            global_step+=1
            

            if (i+1) % 50 == 0:
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader), avg_loss/50.0))
                avg_loss=0.0
            if (i+1)%200==0:
                print("Mask Loss: ",loss_msk)
                print("Depth Loss: ",loss_depth)
            del d0, d1, d2, d3, d4, d5, d6,d0_msk, d1_msk, d2_msk, d3_msk, d4_msk, d5_msk, d6_msk,only_d0, only_d1, only_d2, only_d3, only_d4, only_d5, only_d6,loss2_depth, loss_depth,loss2_msk, loss_msk, loss 
            if (i+1) % 50==0:
                torch.cuda.empty_cache()

            if args.tboard and  (i+1) % 20 == 0:
                #show_wc_tnsboard(global_step, writer,images,labels,pred, 8,'Train Inputs', 'Train Depths', 'Train Pred. Depths')
                writer.add_scalar('Depth: L1 Loss/train', avg_lossall/(i+1), global_step)
                #writer.add_scalar('Depth: Grad Loss/train', avg_gloss/(i+1), global_step)
        #avg_loss=avg_loss/len(trainloader)
        avg_loss2=avg_loss2/len(trainloader)
        #avg_gloss=avg_gloss/len(trainloader)
        print("Training loss2:%4f" %(avg_loss2))
        train_losses=[avg_loss2]

        lrate=get_lr(optimizer)
        write_log_file(log_file_name, train_losses, epoch+1, lrate,'Train')
        

        model.eval() # Tell the model we're evaluating
        val_loss=0.0
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)): #use progress bar
            with torch.no_grad():
                images_val = Variable(images_val.cuda())
                labels_val = Variable(labels_val.cuda())

                d1,d2,d3,d4,d5,d6,d7= model(images_val)

                pred = d1
                pred_val=pred.cpu()
                pred_msk=(torch.sign(relu(pred_val)))
                labels_val=labels_val.cpu()
                labels_msk=(torch.sign(relu(labels_val)))
                msk_index=(labels_msk).nonzero()

                only_d1=d1[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]
                only_d_lbl=labels_val[msk_index[:,0],msk_index[:,1],msk_index[:,2],msk_index[:,3]]


                loss_msk = l1_loss(pred_msk, labels_msk)
                loss_depth=l1_loss(only_d1,only_d_lbl)
                val_loss+=float(loss_msk+loss_depth)
                #val_gloss+=float(g_loss)
                del d1,d2,d3,d4,d5,d6,d7,only_d_lbl,only_d1,pred_msk,labels_msk

                if (i_val+1) % 50==0:
                    torch.cuda.empty_cache()

        if args.tboard:
            #show_wc_tnsboard(epoch+1, writer,images_val,labels_val,pred, 8,'Val Inputs', 'Val Depths', 'Val Pred. Depths')
            writer.add_scalar('Depth: L1 Loss/val', val_loss/len(valloader), epoch+1)
            #writer.add_scalar('Depth: Grad Loss/val', val_gloss, epoch+1)

        val_loss=val_loss/len(valloader)
        print("val loss at epoch {}:: {}".format(epoch+1,val_loss))

        val_losses=[val_loss]
        write_log_file(log_file_name, val_losses, epoch+1, lrate, 'Val')

        #reduce learning rate
        sched.step(val_loss) 
        
        if val_loss < best_val_loss:
            best_val_loss=val_loss
        state = {'epoch': epoch+1,
                    'model_state': model.state_dict(),
                    'optimizer_state' : optimizer.state_dict(),}
        torch.save(state, args.logdir+"{}_{}_{}_{}_{}_best_model.pkl".format(args.arch, epoch+1,avg_loss2,val_loss, experiment_name))
        torch.cuda.empty_cache()
        time.sleep(0.003)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='dnetccnl', 
                        help='Architecture to use [\'dnetccnl, unetnc\']')
    parser.add_argument('--data_path', nargs='?', type=str, default='', 
                        help='Data path to load data')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=200, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=200, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=0.00005, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints-u2depth/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.set_defaults(tboard=True)

    args = parser.parse_args()
    train(args)


# CUDA_VISIBLE_DEVICES=0,1,2 python trainu2net_depth.py --arch u2net_depth --data_path /disk2/sinan/doc3d/ --batch_size 70 --tboard --resume 