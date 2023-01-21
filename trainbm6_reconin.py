# code to train backward mapping regression from GT world coordinates
# models are saved in checkpoints-bm/ 

import sys, os
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
from utils import show_unwarp_tnsboard,  get_lr
import recon_lossc
import pytorch_ssim




def write_log_file(log_file_name,losses, epoch, lrate, phase):
    with open(log_file_name,'a') as f:
        f.write("\n{} LRate: {} Epoch: {} Loss: {} MSE: {}".format(phase, lrate, epoch, losses[0], losses[1]))

def train(args):

    # Setup Dataloader
    data_loader = get_loader('doc3dbm_rgbd_recon_gt')
    data_path = args.data_path
    t_loader = data_loader(data_path, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols))
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=12, shuffle=True,pin_memory=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=12, pin_memory=True)

    # Setup Model
    model = get_model(args.arch, n_classes,in_channels=4, img_size=(args.img_rows, args.img_cols))
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    
    # Optimizer
    optimizer= torch.optim.Adam(model.parameters(),lr=args.l_rate, weight_decay=5e-5, amsgrad=True)

    # LR Scheduler
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Losses
    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss() #For regression loss in terms of the backward mapping function, i.e. the pixel coordinates
    #reconst_loss= recon_lossc.Unwarploss() # The reconstruction loss for the unwarped checkerboard

    epoch_start=0
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    # Log file:
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    experiment_name='bm_6_reconin_ad_192' #network_activation(t=[-1,1])_dataset_lossparams_augmentations_trainstart
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

    best_val_uwarpssim = 99999.0
    best_val_mse=99999.0
    global_step=0

    print(get_lr(optimizer))
    for epoch in range(epoch_start,args.n_epoch):        
        avg_loss=0.0
        avgl1loss=0.0

        avg_loss_bmfromrecon=0.0
        avgl1loss_bmfromrecon=0.0

        avg_loss_ad=0.0
        #avgrloss=0.0
        #avgssimloss=0.0
        train_mse=0.0
        model.train()

        for i, (images,recon, dmap, labels) in enumerate(trainloader):
            rgbd=torch.cat([images,dmap],dim=1)
            rgbd_recon=torch.cat([recon,dmap],dim=1)

            rgbd = Variable(rgbd.cuda())
            rgbd_recon = Variable(rgbd_recon.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()

            target = model(rgbd)
            target_nhwc = target.permute(0,2,3,1)
            l1loss = loss_fn(target_nhwc, labels)

            target_fromrecon=model(rgbd_recon)
            target_nhwc_fromrecon = target_fromrecon.permute(0,2,3,1)
            l1loss_fromrecon = loss_fn(target_nhwc_fromrecon, labels)

            loss_ad=loss_fn(target_nhwc,target_nhwc_fromrecon)            
            #rloss,ssim,uworg,uwpred = reconst_loss(images[:,:-1,:,:],target_nhwc,labels)
            lossall=l1loss+l1loss_fromrecon+loss_ad  #+ (0.3*ssim) +l1loss_fromrecon +l1loss_fromrecon
            # loss=l1loss
            #print(loss_ad)  
            avgl1loss+=float(l1loss)        
            avg_loss+=float(l1loss)

            avgl1loss_bmfromrecon+=float(l1loss_fromrecon)
            avg_loss_bmfromrecon+=float(l1loss_fromrecon)

            avg_loss_ad+=float(loss_ad)
            #avgrloss+=float(rloss)
            #avgssimloss+=float(ssim)
            
            train_mse+=MSE(target_nhwc, labels).item()

            lossall.backward()
            optimizer.step()
            global_step+=1

            if (i+1) % 50 == 0:
                avg_loss=avg_loss/50
                avg_loss_bmfromrecon=avg_loss_bmfromrecon/50
                avg_loss_ad=avg_loss_ad/50
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f Loss From Recon: %.4f Loss AD: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader), avg_loss, avg_loss_bmfromrecon, avg_loss_ad))
                avg_loss=0.0
                avg_loss_bmfromrecon=0.0
                avg_loss_ad=0.0

            if args.tboard and  (i+1) % 20 == 0:
                #show_unwarp_tnsboard(global_step, writer,uwpred,uworg,8,'Train GT unwarp', 'Train Pred Unwarp')
                writer.add_scalar('BM: L1 Loss/train', avgl1loss/(i+1), global_step)
                writer.add_scalar('BM From Recon: L1 Loss/train', avgl1loss_bmfromrecon/(i+1), global_step)
                #writer.add_scalar('CB: Recon Loss/train', avgrloss/(i+1), global_step)
                #writer.add_scalar('CB: SSIM Loss/train', avgssimloss/(i+1), global_step)


        #avgssimloss=avgssimloss/len(trainloader)
        #avgrloss=avgrloss/len(trainloader)
        avgl1loss=avgl1loss/len(trainloader)
        avgl1loss_bmfromrecon=avgl1loss_bmfromrecon/len(trainloader)
        train_mse=train_mse/len(trainloader)
        print("Training L1:%4f" %(avgl1loss))
        print("Training L1 from Recon:%4f" %(avgl1loss_bmfromrecon))
        print("Training MSE:'{}'".format(train_mse))
        train_losses=[avgl1loss, train_mse]
        lrate=get_lr(optimizer)
        write_log_file(log_file_name, train_losses,epoch+1, lrate,'Train')

        #-------------------EVALUATION----------------- 
        model.eval()
        val_loss=0.0
        val_l1loss=0.0
        val_l1loss_fromrecon=0.0
        val_mse=0.0
        #val_rloss=0.0
        #val_ssimloss=0.0

        for i_val, (images,recon,dmap, labels) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                rgbd=torch.cat([images,dmap],dim=1)
                rgbd_recon=torch.cat([recon,dmap],dim=1)

                rgbd = Variable(rgbd.cuda())
                rgbd_recon = Variable(rgbd_recon.cuda())
                labels = Variable(labels.cuda())

                target = model(rgbd)
                target_nhwc = target.permute(0,2,3,1)

                target_fromrecon=model(rgbd_recon)
                target_nhwc_fromrecon = target_fromrecon.permute(0,2,3,1)

                pred=target_nhwc.detach().cpu()
                pred_fromrecon=target_nhwc_fromrecon.detach().cpu()
                gt = labels.cpu()

                l1loss_fromrecon = loss_fn(pred_fromrecon, gt)
                l1loss = loss_fn(pred, gt)
                #rloss,ssim,uworg,uwpred = reconst_loss(images_val[:,:-1,:,:],target_nhwc,labels_val)
                val_l1loss+=float(l1loss)
                val_l1loss_fromrecon+=float(l1loss_fromrecon)
                #val_rloss+=float(rloss.cpu())
                #val_ssimloss+=float(ssim.cpu())
                val_mse+=float(MSE(pred, gt))

        val_l1loss=val_l1loss/len(valloader)
        val_l1loss_fromrecon=val_l1loss_fromrecon/len(valloader)
        val_mse=val_mse/len(valloader)
        #val_ssimloss=val_ssimloss/len(valloader)
        #val_rloss= val_rloss/len(valloader)
        print("val loss at epoch {}:: {}".format(epoch+1,val_l1loss))
        print("val loss from recon at epoch {}:: {}".format(epoch+1,val_l1loss_fromrecon))
        print("val mse: {}".format(val_mse)) 
        val_losses=[val_l1loss, val_mse]
        write_log_file(log_file_name, val_losses, epoch+1, lrate, 'Val')
        if args.tboard:
            # log the val losses
            writer.add_scalar('BM: L1 Loss/val', val_l1loss, epoch+1)
            writer.add_scalar('BM From Recon: L1 Loss/val', val_l1loss_fromrecon, epoch+1)
            #writer.add_scalar('CB: Recon Loss/val', val_rloss, epoch+1)
            #writer.add_scalar('CB: SSIM Loss/val', val_ssimloss, epoch+1)

        #reduce learning rate
        sched.step(val_mse) 

        if val_mse < best_val_mse:
            best_val_mse=val_mse
        state = {'epoch': epoch+1,
                    'model_state': model.state_dict(),
                    'optimizer_state' : optimizer.state_dict(),}
        torch.save(state, args.logdir+"{}_{}_{}_{}_{}_best_model.pkl".format(args.arch, epoch+1,val_mse,train_mse,experiment_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='dnetccnl', 
                        help='Architecture to use [\'dnetccnl, unetnc\']')
    parser.add_argument('--data_path', nargs='?', type=str, default='/disk2/sinan/doc3d/', 
                        help='Data path to load data')
    parser.add_argument('--img_rows', nargs='?', type=int, default=192, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=192, 
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=300, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=115, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-5, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints-bm6_bothrecon_ad_192/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.set_defaults(tboard=True)

    args = parser.parse_args()
    train(args)



#CUDA_VISIBLE_DEVICES=2,0,1 python trainbm6_reconin.py --arch mobilevit_sandfcn_skipRGBD --data_path /disk2/sinan/doc3d/ --img_rows 192 --img_cols 192 --resume /home/sinan/DewarpNet-master/checkpoints-bm6_bothrecon_ad_192/mobilevit_sandfcn_skipRGBD_33_0.00016037944801976025_0.0002287053885697021_bm_6_reconin_ad_192_best_model.pkl