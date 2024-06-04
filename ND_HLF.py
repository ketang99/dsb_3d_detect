import numpy as np
import h5py
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import math
import pandas as pd
import csv
from matplotlib import pyplot as plt
from importlib import import_module
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.init as init
from torchmetrics.classification import BinaryHingeLoss
import sys

epochs = 3
home_dir = '/data/kgupta/registration_testing'
os.chdir(home_dir)
root_dir = home_dir + '/dsb/red_dsb'
ims_dir = home_dir + '/h5_files'
train_filename = 'RED_DSB_trainsplit.h5'
mname = f'test_norm_extractcheck_defanch_sgd_red_dsb_epoch{epochs}_pneq_2patch__e-1'
model_filename = 'singlechannel_red'
model_dir = home_dir + f'/models/{mname}'
model_save_path = f'{model_dir}/epoch_models'
loss_save_path = f'{model_dir}/loss'
sys.path.append(root_dir)
sys.path.append(root_dir + '/training')
sys.path.append(root_dir + '/training/classifier')
sys.path.append(home_dir + '/dsb')
print('Added dirs to path')

if mname not in os.listdir(f'{home_dir}/models'):
    os.mkdir(model_dir)

if 'loss' not in os.listdir(model_dir):
    os.mkdir(loss_save_path)

if 'epoch_models' not in os.listdir(model_dir):
    os.mkdir(model_save_path)

import reg_functions as reg
import data_red_dsb as dsb
from layers import *
import net_detector_3 as nd
# contains the model
import trainval_detector as det
# contains the function that performs training
from config_training import config as config_training
from split_combine import SplitComb

print('DSB modules imported')

from utils import *

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model1', '-m1', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('--model2', '-m2', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-e', '--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-b2', '--batch-size2', default=3, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=None, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='5', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test1', default=0, type=int, metavar='TEST',
                    help='do detection test')
parser.add_argument('--test2', default=0, type=int, metavar='TEST',
                    help='do classifier test')
parser.add_argument('--test3', default=0, type=int, metavar='TEST',
                    help='do classifier test')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=8, type=int, metavar='N',
                    help='number of gpu for test')
parser.add_argument('--debug', default=0, type=int, metavar='TEST',
                    help='debug mode')
parser.add_argument('--freeze_batchnorm', default=0, type=int, metavar='TEST',
                    help='freeze the batchnorm when training')

def get_lr(epoch,args):
    assert epoch<=args.lr_stage[-1]
    if args.lr==None:
        lrstage = np.sum(epoch>args.lr_stage)
        lr = args.lr_preset[lrstage]
    else:
        lr = args.lr
    return lr


def train_nodulenet(data_loader, net, loss, epoch, optimizer, args, epoch_save, model_filename, save_path, loss_path):
    start_time = time.time()
    net.train()
    if args.freeze_batchnorm:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    lr = get_lr(epoch,args)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    raw_metrics = []
   # for (i, batch) in enumerate(data_loader):
    # torch.cuda.empty_cache()
    # iterating through each patch
    # optimizer.zero_grad()
    for (i, batch) in enumerate(data_loader):
        
        print('loader iter: ', i)
       # optimizer.zero_grad()
        data = batch['patch'].to('cuda:0')
        target = batch['label'].to('cuda:0')
        # print('Data length: ', len(data))
        # print('Data shape: ', data.shape)
        # print('Target shape: ', target.shape)

        _, output = net(data)
        loss_output = loss(output, target)
        print('loss_output dims: ', len(loss_output))
        #for t in range(len(loss_output)):
           # print(t, type(loss_output[t]))
           # if isinstance(loss_output[t], torch.Tensor):
            #    print('tensor size: ', loss_output[t].size())
            #print(loss_output[t])

        # print('cls term: ', type(loss_output[1]))
        # torch.tensor(loss_output[1], requires_grad=True).backward()
        
        optimizer.zero_grad()
        
        print('')
        print('loss_output[0]: ', loss_output[0])
        print('loss_output[0] grad fn: ', loss_output[0].grad_fn)
        loss_output[0].backward()
        
        printgrad = False
        large_names = []
        
        if printgrad:
            for name, param in net.named_parameters():
                if param.grad is not None:
                    print(f'{name}.grad: {param.grad.norm()}')
                    if (param.grad.norm()>0.05):
                        large_names.append(name)
                        print('True')

        print('')
        print(large_names)
        print('')

        optimizer.step()

        loss_output[0] = loss_output[0].item()
        # print('loss output val: ', loss_output[0])
        raw_metrics.append(loss_output)

    end_time = time.time()

    ms = []
    # print(metrics.device)
    for i,metric in enumerate(raw_metrics):
        # print(type(metric))
        # print(metric)
        
        m = []
        for v in metric:
            if torch.is_tensor(v):
                m.append(v.item())
            else:
                m.append(v)
        # print(m)
        # print('')
        ms.append(m)


    metricstest = np.asarray(ms[0])
    # print(metricstest)
    metrics = [np.asarray(metric) for metric in ms]
    # print('type metrics before np convert: ', type(metrics))
    metrics = np.asarray(metrics, np.float32)
    # print('metrics: ', metrics)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print('')

    # save the loss metrics
    np.save(f'{loss_path}/Train_loss_metrics_{epoch}.npy', metrics)
    
    # save the parameters
    if epoch_save:
        torch.save(net.state_dict(), f'{save_path}/{model_filename}_{epoch}_state_dict.pth')
        torch.save(net, f'{save_path}/{model_filename}_{epoch}.pth')


def validate_nodulenet(data_loader, net, loss, epoch, args, save_path, loss_path):
    start_time = time.time()
    
    net.eval()

    raw_metrics = []
    for (i, batch) in enumerate(data_loader):
        data = batch['patch'].to('cuda:0')
        target = batch['label'].to('cuda:0')

        _,output = net(data)
        loss_output = loss(output, target, train = False)

        # loss_output[0] = loss_output[0].data[0]
        loss_output[0] = loss_output[0].item()
        raw_metrics.append(loss_output)    
    end_time = time.time()


    ms = []
    # print(metrics.device)
    for i,metric in enumerate(raw_metrics):
        # print(type(metric))
        # print(metric)

        m = []
        for v in metric:
            if torch.is_tensor(v):
                m.append(v.item())
            else:
                m.append(v)
        # print(m)
        # print('')
        ms.append(m)


    metricstest = np.asarray(ms[0])
    # print(metricstest)
    metrics = [np.asarray(metric) for metric in ms]
    metrics = np.asarray(metrics, np.float32)
    print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))

    np.save(f'{loss_path}/Val_loss_metrics_{epoch}.npy', metrics)


def main():
    global args
    args = parser.parse_args()
    torch.manual_seed(0)
    # Call the model
    nodmodel = import_module('net_detector_3')
    config, nod_net, loss, get_pbb = nodmodel.get_model()
    # nod_net = torch.nn.parallel.DistributedDataParallel(nod_net)
    if torch.cuda.is_available():
        nod_net = nod_net.cuda()
    
   # nod_net = torch.nn.DataParallel(nod_net)

    #for layer in nod_net.modules():
        # print(type(layer))
     #   if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.ConvTranspose3d):
            # Initialize Conv3d layers as desired (e.g., He initialization)
      #      init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
       #     if layer.bias is not None:
        #        init.constant_(layer.bias, 0)  # Initialize biases to zeros
        # elif isinstance(layer, nn.BatchNorm3d):
        #     # Initialize BatchNorm3d layers as desired (e.g., Xavier initialization)
        #     init.xavier_uniform_(layer.weight)
        #     init.constant_(layer.bias, 0)  # Initialize biases to zeros
        # Add more conditions for other layer types as needed
    
    print('Inits done')
    
    # config['anchors'] = [50.0,55.0,60.0,65.0]
    config['anchors'] = [10.0, 30.0, 60.0]
    config['lr_stage'] = [50,100,150,250]
    print('config: ')
    print(config)

    print('Model retrieved and parallelized')
    #return
    # Initialize the parameters / call them from a checkpoint
    # checkpoint = torch.load(args.resume) # try to see where args.resume points to. Do this if that pointed location exists
    
    # Call the optimizer (SGD)
    optimizer = torch.optim.SGD(nod_net.parameters(),args.lr,momentum = 0.9,weight_decay = args.weight_decay)
    # optimizer = torch.optim.Adagrad(nod_net.parameters(), args.lr)
    # optimizer = torch.optim.Adam(nod_net.parameters(), args.lr)
    split_comber = SplitComb(192, config['max_stride'], config['stride'], 32, 0)
    # Call data object and dataloader from anchor_loss (dsb)
    train_dataset = dsb.HLFBoneMarrowCells(train_filename, ims_dir, config, split_comber, phase='Train')
    val_dataset = dsb.HLFBoneMarrowCells(train_filename, ims_dir, config, split_comber, phase='Val')
    print(f'batch size = {args.batch_size}')
    print(f'learning rate = {args.lr}')
    print(f'weight decay: {args.weight_decay}')
   # return
    train_loader_nod = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dsb.custom_collate_dict_new)
    val_loader_nod = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dsb.custom_collate_dict_new)
    print('Train and val loaders initialized')

    print('Freeze batchnorm? ', args.freeze_batchnorm)
    if args.freeze_batchnorm:
        print('Freeze is true, make it false to continue')
        return

    args.lr_stage = config['lr_stage']

    print('Entering training epochs')
    print('')
    print(f'Epochs = {epochs}')
    epochs_to_save = range(epochs)
    start_time = time.time()
    for i in range(epochs):
        if i in epochs_to_save:
            epoch_save = True
        else:
            epoch_save = False
        print('************************')
        print(f'Training epoch {i}')
        nod_net.train()
        train_nodulenet(train_loader_nod, nod_net, loss, i, optimizer, args, epoch_save, model_filename, model_save_path, loss_save_path)
        print(f'Training for epoch {i} is complete')
        print('***')
        print('Validation:')
        validate_nodulenet(val_loader_nod, nod_net, loss, i, args, model_save_path, loss_save_path)
        print(f'Validation for epoch {i} is complete')
        print('************************')

    print('\n\n')
    print('*   *   *   *   *')
    print(f'All {epochs}  epochs complete')
    print(f'Time taken = {time.time()-start_time}')

if __name__ == '__main__':
    main()
