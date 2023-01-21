# code to train world coord regression from RGB Image
# models are saved in checkpoints-wc/

import sys, os
from turtle import forward
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


def write_log_file(log_file_name,losses, epoch, lrate, phase):
    with open(log_file_name,'a') as f:
        f.write("\n{} LRate: {} Epoch: {} Loss: {} MSE: {}".format(phase, lrate, epoch, losses[0], losses[1]))
        




def train(args):

    # Setup Dataloader
    data_loader = get_loader('doc3dboundary')
    data_path = args.data_path
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=True)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Model
    model = get_model(args.arch, n_classes,in_channels=3) #Get the U-Net architecture
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    # Activation
    activation=nn.Sigmoid()
    
    # Optimizer
    optimizer= torch.optim.Adam(model.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)

    # LR Scheduler, which can reduce the learning rate as time passing by
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Losses
    MSE = nn.MSELoss() #L2 Loss for measure the training loss and validation loss
    loss_fn = nn.BCELoss() # BCE Loss used throughout the whole training process, including L_C, L_D, L_T (Both in the first and second training phase)
    #gloss= grad_loss.Gradloss(window_size=5,padding=2) #The Gradient Loss used in L_C, i.e. deltaChead-deltaC

    epoch_start=0
    if args.resume is not None:                                         
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
    experiment_name='boundary' #activation_dataset_lossparams_augmentations_trainstart
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
    print(get_lr(optimizer))

    for epoch in range(epoch_start,args.n_epoch):
        avg_loss=0.0
        avg_l1loss=0.0
        #avg_gloss=0.0
        train_mse=0.0
        model.train() # Tell the model we're training

        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)
            pred=activation(outputs) #Activation Function
            #g_loss=gloss(pred, labels) # The Gradient Loss between the Ground Truth 3D representation and the pedicted one
            l1loss = loss_fn(pred, labels) # The BCE Loss between the above two
            loss=l1loss#i.e. the L_C loss, actually I uncomment this to make this consistent with the paper, originally the latter part is commented, which I don't know the reason now.
            avg_l1loss+=float(l1loss)
            #avg_gloss+=float(g_loss)
            avg_loss+=float(loss)
            train_mse+=float(MSE(pred, labels).item())

            loss.backward()
            optimizer.step()
            global_step+=1

            if (i+1) % 50 == 0:
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader), avg_loss/50.0))
                avg_loss=0.0

            if args.tboard and  (i+1) % 20 == 0:
                #show_wc_tnsboard(global_step, writer,images,labels,pred, 8,'Train Inputs', 'Train Boundarys', 'Train Pred. Boundarys')
                writer.add_scalar('Boundary: BCE Loss/train', avg_l1loss/(i+1), global_step)
                #writer.add_scalar('Boundary: Grad Loss/train', avg_gloss/(i+1), global_step)
                
        train_mse=train_mse/len(trainloader)
        avg_l1loss=avg_l1loss/len(trainloader)
        #avg_gloss=avg_gloss/len(trainloader)
        print("Training BCE:%4f" %(avg_l1loss))
        print("Training MSE:'{}'".format(train_mse))
        train_losses=[avg_l1loss, train_mse]

        lrate=get_lr(optimizer)
        write_log_file(experiment_name, train_losses, epoch+1, lrate,'Train')
        

        model.eval() # Tell the model we're evaluating
        val_loss=0.0
        val_mse=0.0
        val_bg=0.0
        val_fg=0.0
        val_gloss=0.0
        val_dloss=0.0
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)): #use progress bar
            with torch.no_grad():
                images_val = Variable(images_val.cuda())
                labels_val = Variable(labels_val.cuda())
                
                outputs = model(images_val)
                pred_val=activation(outputs)
                #g_loss=gloss(pred_val, labels_val).cpu()
                pred_val=pred_val.cpu()
                labels_val=labels_val.cpu()
                loss = loss_fn(pred_val, labels_val)
                val_loss+=float(loss)
                val_mse+=float(MSE(pred_val, labels_val))
                #val_gloss+=float(g_loss)

        if args.tboard:
            #show_wc_tnsboard(epoch+1, writer,images_val,labels_val,pred, 8,'Val Inputs', 'Val Boundarys', 'Val Pred. Boundarys')
            writer.add_scalar('Boundary: BCE Loss/val', val_loss, epoch+1)
            #writer.add_scalar('Boundary: Grad Loss/val', val_gloss, epoch+1)

        val_loss=val_loss/len(valloader)
        val_mse=val_mse/len(valloader)
        val_gloss=val_gloss/len(valloader)
        print("val loss at epoch {}:: {}".format(epoch+1,val_loss))
        print("val MSE: {}".format(val_mse))

        val_losses=[val_loss, val_mse]
        write_log_file(experiment_name, val_losses, epoch+1, lrate, 'Val')

        #reduce learning rate
        sched.step(val_loss) 
        
        if val_loss < best_val_loss:
            best_val_loss=val_loss
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, args.logdir+"{}_{}_{}_{}_{}_best_model.pkl".format(args.arch, epoch+1,val_mse,train_mse,experiment_name))

        if (epoch+1) % 10 == 0:
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, args.logdir+"{}_{}_{}_{}_{}_model.pkl".format(args.arch, epoch+1,val_mse,train_mse,experiment_name))
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
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints-squeezeboundary/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.set_defaults(tboard=True)

    args = parser.parse_args()
    train(args)


# CUDA_VISIBLE_DEVICES=0,1,2 python trainboundary.py --arch squeezeunet --data_path /disk2/sinan/doc3d/ --batch_size 200 --tboard --resume /home/sinan/DewarpNet-master/checkpoints-squeezeboundary/squeezeunet_150_0.007575992671772838_0.006176567144506083_boundary_model.pkl