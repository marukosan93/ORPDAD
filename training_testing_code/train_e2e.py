'''
training on the speficed end-to-end method and fold and scenarios
Saves pth model and losses
'''

import argparse
import os
import pickle
import time
from utils_dl import SNR_loss
import more_itertools as mit
import numpy as np
from torch.utils import tensorboard
import torch
import torch.nn as nn
import torch.optim as op
import torchvision.transforms as T
from scipy.signal import welch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from scipy import signal
from block_loader import block
from block_loader_deepphys import block_deepphys
from models.DeepPhys import DeepPhys
from models.TSCAN import TSCAN
from models.EfficientPhys import EfficientPhys
from models.PhysNet import PhysNet
from models.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from models.contrastphys.PhysNetModel import PhysNetC
from models.contrastphys.loss import ContrastLoss
from models.contrastphys.IrrelevantPowerRatio import IrrelevantPowerRatio
from models.contrastphys.utils_data import *
from models.contrastphys.utils_sig import *
from utils_dl import setup_seed, AverageMeter, Acc, NegativeMaxCrossCorr, MapPSD, concatenate_output, set_scenario,NegPearson,split_clips
from utils_trad import butter_bandpass, calc_hr
from models.TorchLossComputer import TorchLossComputer
import math
import matplotlib.pyplot as plt


f_min = 0.5
f_max = 3

#initialise to 0, then assign for each method
alpha = 0
gamma = 0
delta = 0

def train(train_loader, model, criterion_hr, criterion_rppg, criterion_fft, optimizer, epoch):

    #Run one train epoch
    fps = 30
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_hr = AverageMeter()
    losses_rppg = AverageMeter()
    losses_fft = AverageMeter()

    # switch to train mode
    model.train()
    IPR = IrrelevantPowerRatio(Fs=fps, high_pass=40, low_pass=250)
    end = time.time()
    for i,  (input,bvp,hr_bvp) in enumerate(train_loader):
        # measur data loading time
        data_time.update(time.time() - end)

        input = input.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)
        hr_bvp = hr_bvp.to(device)

        if method == "tscan" or method == "efficientphys" or method == "deepphys":
            input_reshaped = input
            if method == "efficientphys":
                input_reshaped = input_reshaped[:,:,3:,:,:]
            b,t,c,h,w = input_reshaped.size()

            input_reshaped = input_reshaped.view(b * t, c, h, w)
            bvp = bvp.view(-1, 1)
            input_reshaped = input_reshaped[:(b * t) // 10*10]
            bvp = bvp[:(b * t) // 10*10]
            if method == "efficientphys":
                last_frame = torch.unsqueeze(input_reshaped[-1, :, :, :], 0).repeat(1, 1, 1, 1)
                input_reshaped = torch.cat((input_reshaped, last_frame), 0)

            output = model(input_reshaped)
            output = output.permute(1,0)
            bvp = bvp.permute(1,0)
            bvp = bvp.view(b,t)
            output = output.view((b,t))

        if method == "physnet":
            output = model(input)
            output = (output-torch.mean(output)) /torch.std(output)
            bvp = (bvp-torch.mean(bvp)) /torch.std(bvp)
        if method == "physformer":
            output = model(input,2.0)
            output = (output-torch.mean(output)) /torch.std(output)
            bvp = (bvp-torch.mean(bvp)) /torch.std(bvp)
            hr_bvp  = torch.abs(hr_bvp - 40)
        if method == "contrastphys":
            model_output = model(input)
            output = model_output[:,-1]

        if method == "tscan" or method == "efficientphys" or method == "deepphys":
            loss_rppg = criterion_rppg(output.unsqueeze(1).unsqueeze(1),bvp.unsqueeze(1).unsqueeze(1))
            loss_hr = loss_rppg
            loss_fft = loss_rppg
        #Losses
        if method == "physnet":
            loss_rppg = criterion_rppg(output.unsqueeze(1).unsqueeze(1),bvp.unsqueeze(1).unsqueeze(1))
            loss_hr = loss_rppg
            loss_fft = loss_rppg
        if method == "physformer":
            loss_rppg = math.pow(3, epoch/50)*criterion_rppg(output.unsqueeze(1).unsqueeze(1),bvp.unsqueeze(1).unsqueeze(1))
            fre_loss = 0.0
            kl_loss = 0.0
            train_mae = 0.0
            for bb in range(input.shape[0]):
                loss_distribution_kl, fre_loss_temp, train_mae_temp = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(output[bb], hr_bvp[bb], 30, std=1.0)  # std=1.1
                fre_loss = fre_loss + fre_loss_temp
                kl_loss = kl_loss + loss_distribution_kl
                train_mae = train_mae + train_mae_temp
            fre_loss = fre_loss/input.shape[0]
            kl_loss = kl_loss/input.shape[0]
            train_mae = train_mae/input.shape[0]
            loss_hr = kl_loss
            loss_fft = fre_loss
        if method == "contrastphys":
            loss_rppg, p_loss, n_loss = criterion_rppg(model_output)
            loss_hr = loss_rppg
            loss_fft = loss_rppg

        loss = alpha * loss_hr + delta * loss_fft + gamma * loss_rppg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        if method == "contrastphys": # we use ipr here to check the network performance and for stopping criteria in validation
            ipr = torch.mean(IPR(output.clone().detach()))
            loss = ipr

        losses.update(loss.item(), input.size(0))
        losses_hr.update(loss_hr.item(), input.size(0))
        losses_rppg.update(loss_rppg.item(), input.size(0))
        losses_fft.update(loss_fft.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 or i == len(train_loader) - 1:
            fig,ax = plt.subplots(1,1)
            ax.plot(output[:,0].detach().cpu().numpy())
            ax.plot(bvp[:,0].detach().cpu().numpy())
            plt.savefig("plottting_e"+str(epoch)+"_i"+str(i)+".png")
            plt.close()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_hr {loss_hr.val:.4f} ({loss_hr.avg:.4f})\t'
                  'Loss_rppg {loss_rppg.val:.4f} ({loss_rppg.avg:.4f})\t'
                  'Loss_fft {loss_fft.val:.4f} ({loss_fft.avg:.4f})\t'
                  'Loss {loss.val:.4f} (loss.avg:.4f)\n'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss_hr=losses_hr, loss_rppg=losses_rppg, loss_fft=losses_fft, loss=losses))
    return losses.avg, losses_rppg.avg, losses_hr.avg, losses_fft.avg

# we need to record all outputs and then return them
def validate(valid_loader, model, epoch):
    # Run one train epoch
    fps = 30
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_hr = AverageMeter()
    losses_rppg = AverageMeter()
    losses_fft = AverageMeter()
    # switch to train mode
    model.eval()
    IPR = IrrelevantPowerRatio(Fs=fps, high_pass=40, low_pass=250)
    end = time.time()
    for i, (input,bvp,hr_bvp) in enumerate(valid_loader):
        # measur data loading time
        data_time.update(time.time() - end)

        input = input.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)
        hr_bvp = hr_bvp.to(device)

        # out_hr, output, feat = model(mstmap)
        with torch.no_grad():
            if method == "tscan" or method == "efficientphys" or method == "deepphys":
                input_reshaped = input
                if method == "efficientphys":
                    input_reshaped = input_reshaped[:,:,3:,:,:]
                b,t,c,h,w = input_reshaped.size()

                input_reshaped = input_reshaped.view(b * t, c, h, w)
                bvp = bvp.view(-1, 1)
                input_reshaped = input_reshaped[:(b * t) // 10*10]
                bvp = bvp[:(b * t) // 10*10]
                if method == "efficientphys":
                    last_frame = torch.unsqueeze(input_reshaped[-1, :, :, :], 0).repeat(1, 1, 1, 1)
                    input_reshaped = torch.cat((input_reshaped, last_frame), 0)

                output = model(input_reshaped)
                output = output.permute(1,0)
                bvp = bvp.permute(1,0)
            if method == "physnet":
                output = model(input)
                output = (output-torch.mean(output)) /torch.std(output)
                bvp = (bvp-torch.mean(bvp)) /torch.std(bvp)
            if method == "physformer":
                output = model(input,2.0)
                output = (output-torch.mean(output)) /torch.std(output)
                bvp = (bvp-torch.mean(bvp)) /torch.std(bvp)
                hr_bvp  = torch.abs(hr_bvp - 40)
            if method == "contrastphys":
                model_output = model(input)
                output = model_output[:,-1]

        if method == "tscan" or method == "efficientphys" or method == "deepphys":
            loss_rppg = criterion_rppg(output.unsqueeze(1).unsqueeze(1),bvp.unsqueeze(1).unsqueeze(1))
            loss_hr = loss_rppg
            loss_fft = loss_rppg

        if method == "physnet":
            loss_rppg = criterion_rppg(output.unsqueeze(1).unsqueeze(1),bvp.unsqueeze(1).unsqueeze(1))
            loss_hr = loss_rppg
            loss_fft = loss_rppg
        if method == "physformer":
            loss_rppg = math.pow(3, epoch/50)*criterion_rppg(output.unsqueeze(1).unsqueeze(1),bvp.unsqueeze(1).unsqueeze(1))
            fre_loss = 0.0
            kl_loss = 0.0
            train_mae = 0.0
            for bb in range(input.shape[0]):
                loss_distribution_kl, fre_loss_temp, train_mae_temp = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(output[bb], hr_bvp[bb], 30, std=1.0)  # std=1.1
                fre_loss = fre_loss + fre_loss_temp
                kl_loss = kl_loss + loss_distribution_kl
                train_mae = train_mae + train_mae_temp
            fre_loss = fre_loss/input.shape[0]
            kl_loss = kl_loss/input.shape[0]
            train_mae = train_mae/input.shape[0]
            loss_hr = kl_loss
            loss_fft = fre_loss
        if method == "contrastphys":
            loss_rppg, p_loss, n_loss = criterion_rppg(model_output)
            loss_hr = loss_rppg
            loss_fft = loss_rppg


        loss = alpha * loss_hr + delta * loss_fft + gamma * loss_rppg

        loss = loss.float()

        if method == "contrastphys": # we use ipr here to check the network performance and for stopping criteria in validation
            ipr = torch.mean(IPR(output.clone().detach()))
            loss = ipr

        losses.update(loss.item(), input.size(0))
        losses_hr.update(loss_hr.item(), input.size(0))
        losses_rppg.update(loss_rppg.item(), input.size(0))
        losses_fft.update(loss_fft.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0 or i == len(valid_loader) - 1:
            # print(gt_hr*140+40)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_hr {loss_hr.val:.4f} ({loss_hr.avg:.4f})\t'
                  'Loss_rppg {loss_rppg.val:.4f} ({loss_rppg.avg:.4f})\t'
                  'Loss_fft {loss_fft.val:.4f} ({loss_fft.avg:.4f})\t'
                  'Loss {loss.val:.4f} (loss.avg:.4f)\n'.format(
                epoch, i, len(valid_loader), batch_time=batch_time,
                data_time=data_time, loss_hr=losses_hr, loss_rppg=losses_rppg, loss_fft=losses_fft, loss=losses))
    return losses.avg, losses_rppg.avg, losses_hr.avg, losses_fft.avg

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fold', type=str, required=True)
parser.add_argument('-m', '--method', type=str, required=True)
parser.add_argument('-ts','--train_set', type=str, required=True,
                    help='input for multiple scenarios should be like this: still+illumination, input for single scenario should be like: still')

args = parser.parse_args()

setup_seed()
method = args.method

if method == "physnet" or method == "physformer":
    train_stride = 128
    seq_len = 128
if method == "contrastphys":
    train_stride = 300
    seq_len = 300
if method == "tscan" or method == "efficientphys" or method == "deepphys":
    frame_depth = 10
    train_stride = 180#frame_depth
    seq_len = 180#frame_depth#+1

if method == "physnet":
    alpha = 0 #SHOULD STAY 0, PHYSNET does not have this loss term
    gamma = 1
    delta = 0 #SHOULD STAY 0, PHYSNET does not have this loss term
if method == "physformer":   #can overfit easily
    alpha = 1
    gamma = 0.1
    delta = 1
if method == "contrastphys":
    alpha = 0 #SHOULD STAY 0, CONTRASTPHYS does not have this loss term
    gamma = 1
    delta = 0 #SHOULD STAY 0, CONTRASTPHYS does not have this loss term

if method == "tscan" or method == "efficientphys" or method == "deepphys":
    alpha = 0
    gamma = 1
    delta = 0

if method == "tscan" or method == "efficientphys" or method == "deepphys":
    BATCH_SIZE = 4
if method == "physnet":
    BATCH_SIZE = 8
if method == "physformer":
    BATCH_SIZE = 8
if method == "contrastphys":
    BATCH_SIZE = 2

NUM_WORKERS = 2*BATCH_SIZE
if NUM_WORKERS > 10:
    NUM_WORKERS = 10

train_scen = set_scenario(args.train_set)
valid_scen = train_scen

fold = int(args.fold) - 1 # folds go from 1 to 5
total_subject_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030']

divided = ([list(x) for x in mit.divide(5, total_subject_list)])

train_div = list(np.arange(0, 5))
train_div.remove(fold)

train_subj = [*divided[train_div[0]], *divided[train_div[1]], *divided[train_div[2]], *divided[train_div[3]]] #train
valid_subj = divided[fold]  #validate

train_set = split_clips(train_subj, train_scen, train_stride, seq_len)
valid_set = split_clips(valid_subj, valid_scen, seq_len, seq_len)

if method == "tscan" or method == "efficientphys" or method == "deepphys":
    resize_size=72#36
else:
    resize_size=128

if method == "tscan" or method == "efficientphys" or method == "deepphys":
    train_dataset = block_deepphys(data=train_set,stride=train_stride,shuffle=True, resize_size=resize_size,seq_len=seq_len)
    valid_dataset = block_deepphys(data=valid_set,stride=seq_len,shuffle=False, resize_size=resize_size,seq_len=seq_len)
else:
    train_dataset = block(data=train_set,stride=train_stride,shuffle=True, resize_size=resize_size,seq_len=seq_len)
    valid_dataset = block(data=valid_set,stride=seq_len,shuffle=False, resize_size=resize_size,seq_len=seq_len)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=False)

if method == "deepphys":
    model = DeepPhys(img_size=resize_size)#frame_depth, 32, 64, (resize_size, resize_size, 3))(img_size=resize_size)
if method == "tscan":
    model = TSCAN(frame_depth=frame_depth,img_size=resize_size)#frame_depth, 32, 64, (resize_size, resize_size, 3))(img_size=resize_size)
if method == "efficientphys":
    model = EfficientPhys(frame_depth=frame_depth,img_size=resize_size)#frame_depth, 32, 64, (resize_size, resize_size, 3))(img_size=resize_size)
if method == "physnet":
    model = PhysNet(seq_len)
if method == "physformer":
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(seq_len,128,128), patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
if method == "contrastphys":
    model = PhysNetC()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)

model.to(device)


criterion_hr = nn.L1Loss()
criterion_hr = criterion_hr.to(device)
if method == "tscan" or method == "efficientphys" or method == "deepphys":
    criterion_rppg = NegativeMaxCrossCorr(180,42)
    criterion_fft = MapPSD("mse")
if method == "physnet":
    criterion_rppg =  NegativeMaxCrossCorr(180,42)
    criterion_fft = MapPSD("mse")
if method == "physformer":
    criterion_rppg = NegativeMaxCrossCorr(180,42)
    criterion_fft = MapPSD("mse")
if method == "contrastphys":
    criterion_rppg = ContrastLoss(150, 4, 30, high_pass=40, low_pass=250)
    criterion_fft = MapPSD("mse")

criterion_fft = criterion_fft.to(device)
criterion_rppg = criterion_rppg.to(device)

if method == "tscan" or method == "efficientphys" or method == "deepphys":
    optimizer = op.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=5e-5, weight_decay=0.05)
if method == "physnet":
    optimizer = op.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=5e-5, weight_decay=0.05)
if method == "physformer":
    optimizer = op.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=5e-5, weight_decay=0.05)
if method == "contrastphys":
    optimizer = op.AdamW(model.parameters(), lr=5e-5)

total_epochs = 50

if __name__ == '__main__':
    train_dir, valid_dir = "", ""
    if args.train_set.__contains__("+"):
        train_list = args.train_set.split("+")
        for i in range(len(train_list)):
            train_dir = train_dir + train_list[i]
            if not i == len(train_list) - 1:
                train_dir = train_dir + "_"
    else:
        train_dir = args.train_set

    if args.train_set.__contains__("+"):
        valid_list = args.train_set.split("+")
        for i in range(len(valid_list)):
            valid_dir = valid_dir + valid_list[i]
            if not i == len(valid_list) - 1:
                valid_dir = valid_dir + "_"
    else:
        valid_dir = args.train_set


    logdir = './records/logs/' + method + '__' + train_dir + '__' + args.fold
    model_saved_path = './records/model/' + method + '__' + train_dir + '__' + args.fold + '/'

    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = {
        'loss_train': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_train')),
        'loss_rppg_train': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_rppg_train')),
        'loss_hr_train': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_hr_train')),
        'loss_fft_train': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_fft_train')),
        'loss_valid': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_valid')),
        'loss_rppg_valid': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_rppg_valid')),
        'loss_hr_valid': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_hr_valid')),
        'loss_fft_valid': tensorboard.SummaryWriter(os.path.join(logdir, 'loss_fft_valid'))
    }

    print('start training...')

    for epoch in range(0, total_epochs):
        if method == "contrastphys":
            train_dataset = block(data=train_set,stride=train_stride,shuffle=True, resize_size=resize_size,seq_len=seq_len)
            train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
        losses_train, losses_rppg_train, losses_hr_train, losses_fft_train \
            = train(train_loader, model, criterion_hr, criterion_rppg, criterion_fft, optimizer, epoch)
        losses_valid, losses_rppg_valid, losses_hr_valid, losses_fft_valid \
            = validate(valid_loader, model, epoch)
        torch.save(model.state_dict(), os.path.join(model_saved_path, 'last.pth'))

        writer['loss_train'].add_scalar("loss", losses_train, epoch)
        writer['loss_valid'].add_scalar("loss", losses_valid, epoch)
        writer['loss_rppg_train'].add_scalar("loss_rppg", losses_rppg_train, epoch)
        writer['loss_rppg_valid'].add_scalar("loss_rppg", losses_rppg_valid, epoch)
        writer['loss_hr_train'].add_scalar("loss_hr", losses_hr_train, epoch)
        writer['loss_hr_valid'].add_scalar("loss_hr", losses_hr_valid, epoch)
        writer['loss_fft_train'].add_scalar("loss_fft", losses_fft_train, epoch)
        writer['loss_fft_valid'].add_scalar("loss_fft", losses_fft_valid, epoch)

    # here to save the model and losses
    writer['loss_train'].close()
    writer['loss_valid'].close()
    writer['loss_rppg_train'].close()
    writer['loss_rppg_valid'].close()
    writer['loss_hr_train'].close()
    writer['loss_hr_valid'].close()
    writer['loss_fft_train'].close()
    writer['loss_fft_valid'].close()

    print('finished training')
