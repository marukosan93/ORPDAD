'''
training on the speficed non-end-to-end method and fold and scenarios
Saves pth model and losses
'''

import argparse
import os
import pickle
import time
import more_itertools as mit
import numpy as np
import tensorboard_logger as tb_logger
from torch.utils import tensorboard
import torch
import torch.nn as nn
import torch.optim as op
import torchvision.transforms as T
from scipy.signal import welch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from scipy import signal
from MSTmap_loader import mst
from models.BVPNet import BVPNet
# from torchsummary import summary
from models.model_disentangle import HR_estimator_multi_task_STmap
from models.swin_transformer_unet_skip_expand_decoder_sys_nosq import SwinTransformerSys
from utils_dl import setup_seed, AverageMeter, NegativeMaxCrossCorr, MapPSD, concatenate_output, set_scenario,NegPearson,MapPearson,SNR_loss, split_clips
from utils_trad import butter_bandpass, calc_hr

f_min = 0.5
f_max = 3

#initialise to 0, then assign for each method
alpha = 0# 1
gamma = 0# 1
delta = 0#20#20

method = "" #so that it's a global variable?

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

    end = time.time()
    for i, (mstmap,yuv_mstmap,bvpmap,bvp,hr_bvp) in enumerate(train_loader):
        # measur data loading time
        data_time.update(time.time() - end)

        mstmap = mstmap.to(device=device, dtype=torch.float)

        bvpmap = bvpmap.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)
        yuv_mstmap = yuv_mstmap.to(device=device, dtype=torch.float)
        rgbyuv_mstmap = torch.cat([mstmap,yuv_mstmap],dim=1)

        hr_bvp = hr_bvp.to(device)

        #Forward pass
        if method == "physunet":
            output, out_hr, feat = model(mstmap)
        if method == "bvpnet":
            output, feat = model(yuv_mstmap)
        if method == "cvd":
            out_hr, output, feat = model(rgbyuv_mstmap)

        #Losses
        if method == "bvpnet" or method =="physunet":
            loss_rppg = criterion_rppg(output,bvpmap)
            loss_fft = criterion_fft(output,bvpmap,fps,f_min,f_max)
        if method == "cvd":
            loss_rppg = criterion_rppg(output.unsqueeze(1).unsqueeze(1),bvp.unsqueeze(1).unsqueeze(1))
            loss_fft = criterion_fft(output.unsqueeze(1).unsqueeze(1),bvp.unsqueeze(1).unsqueeze(1),fps,f_min,f_max)
        if method == "bvpnet":
            loss_hr = loss_rppg
        if method == "physunet" or method == "cvd":
            predict = out_hr.squeeze()
            target = hr_bvp
            predict = (predict - 40) / 140
            target = (target-40) / 140
            loss_hr = criterion_hr(predict, target)

        loss = alpha * loss_hr + delta * loss_fft + gamma * loss_rppg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        losses.update(loss.item(), mstmap.size(0))
        losses_hr.update(loss_hr.item(), mstmap.size(0))
        losses_rppg.update(loss_rppg.item(), mstmap.size(0))
        losses_fft.update(loss_fft.item(), mstmap.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0 or i == len(train_loader) - 1:
            #print(gt_hr*140+40)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_hr {loss_hr.val:.4f} ({loss_hr.avg:.4f})\t'
                  'Loss_rppg {loss_rppg.val:.4f} ({loss_rppg.avg:.4f})\t'
                  'Loss_fft {loss_fft.val:.4f} ({loss_fft.avg:.4f})\t'
                  'Loss {loss.val:.4f} (loss.avg:.4f)\n'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss_hr=losses_hr, loss_rppg=losses_rppg, loss_fft=losses_fft, loss=losses))
    losses = losses_rppg
    return losses.avg, losses_rppg.avg, losses_hr.avg, losses_fft.avg

def validate(valid_loader, model, epoch):
    fps = 30
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_hr = AverageMeter()
    losses_rppg = AverageMeter()
    losses_fft = AverageMeter()

    model.eval()

    end = time.time()
    for i, (mstmap, yuv_mstmap, bvpmap, bvp,hr_bvp) in enumerate(valid_loader):
        # measur data loading time
        data_time.update(time.time() - end)

        mstmap = mstmap.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)
        yuv_mstmap = yuv_mstmap.to(device=device, dtype=torch.float)
        rgbyuv_mstmap = torch.cat([mstmap,yuv_mstmap],dim=1)

        hr_bvp =hr_bvp.to(device)

        #Inference
        with torch.no_grad():
            if method == "physunet":
                output, out_hr, feat = model(mstmap)
            if method == "bvpnet":
                output, feat  = model(yuv_mstmap)
            if method == "cvd":
                out_hr, output, feat = model(rgbyuv_mstmap)

        #Losses
        if method == "bvpnet" or method =="physunet":
            loss_rppg = criterion_rppg(output,bvpmap)
            loss_fft = criterion_fft(output,bvpmap,fps,f_min,f_max)
        if method == "cvd":
            loss_rppg = criterion_rppg(output.unsqueeze(1).unsqueeze(1),bvp.unsqueeze(1).unsqueeze(1))
            loss_fft = criterion_fft(output.unsqueeze(1).unsqueeze(1),bvp.unsqueeze(1).unsqueeze(1),fps,f_min,f_max)#, tmp = criterion_fft(output, hr_bvp.view(-1,1), (torch.ones_like(hr_bvp)*fps).view(-1,1), pred = output, flag = None)

        if method == "bvpnet":
            loss_hr = loss_rppg
        if method == "physunet" or method == "cvd":
            predict = out_hr.squeeze()
            target = hr_bvp
            predict = (predict - 40) / 140
            target = (target-40) / 140
            loss_hr = criterion_hr(predict, target)

        loss = alpha * loss_hr + delta * loss_fft + gamma * loss_rppg

        loss = loss.float()

        losses.update(loss.item(), mstmap.size(0))
        losses_hr.update(loss_hr.item(), mstmap.size(0))
        losses_rppg.update(loss_rppg.item(), mstmap.size(0))
        losses_fft.update(loss_fft.item(), mstmap.size(0))

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
    losses = losses_rppg
    return losses.avg, losses_rppg.avg, losses_hr.avg, losses_fft.avg



parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fold', type=str, required=True)
parser.add_argument('-m', '--method', type=str, required=True)
parser.add_argument('-ts','--train_set', type=str, required=True,
                    help='input for multiple scenarios should be like this: still+illumination, input for single scenario should be like: still')
args = parser.parse_args()

setup_seed()
method = args.method

train_stride = 60

if method == "physunet":
    seq_len = 576
if method == "cvd":
    seq_len = 300
if method == "bvpnet":
    seq_len = 256

if method == "physunet":
    alpha = 5#5
    gamma = 1
    delta = 5#5
if method == "cvd":
    alpha = 1#1
    gamma = 1
    delta = 5#5
if method == "bvpnet":
    alpha = 0 #SHOULD STAY 0, BVPNET does not have this loss term
    gamma = 1
    delta = 1#1

if method == "physunet":
    BATCH_SIZE = 8
if method == "bvpnet":
    BATCH_SIZE = 8
if method == "cvd":
    BATCH_SIZE = 8
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

if method == "physunet" or method == "bvpnet":
    transforms = [T.ToTensor(), T.Resize((64, seq_len))]
if method == "cvd":
    transforms = [T.ToTensor(), T.Resize((320, 320))]

transforms = T.Compose(transforms)
train_dataset = mst(data=train_set, stride=train_stride, shuffle=True, transform=transforms, seq_len=seq_len)
valid_dataset = mst(data=valid_set, stride=seq_len, shuffle=False, transform=transforms, seq_len=seq_len)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=False)

if method == "cvd":
    model = HR_estimator_multi_task_STmap(video_length = 300)
if method == "physunet":   # since this ones T=576 is 19.2s, for validation we just put together the 3 segments from the total 70s, and have two clips that are slightly shorter than 30s, it's ok
    model = SwinTransformerSys(img_size=(64,576),
                                    patch_size=4,
                                    in_chans=3,
                                    num_classes=3,
                                    embed_dim=96,
                                    depths=[2, 2, 2, 2],
                                    depths_decoder=[1, 2, 2, 2],
                                    num_heads=[3,6,12,24],
                                    window_size=4,
                                    mlp_ratio=2,
                                    qkv_bias=True,
                                    qk_scale=None,
                                    drop_rate=0,
                                    drop_path_rate=0,
                                    ape=False,
                                    patch_norm=True,
                                    use_checkpoint=False)
if method == "bvpnet":
    model = BVPNet(frames=256,fw=48)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
model.to(device)

criterion_hr = nn.L1Loss()
criterion_hr = criterion_hr.to(device)
if method == "physunet":
    criterion_rppg = NegativeMaxCrossCorr(180,42)
    criterion_fft = MapPSD("mse")
if method == "bvpnet":
    criterion_rppg = NegativeMaxCrossCorr(180,42)
    criterion_fft = MapPSD("l1")#MapPSD("l1")
if method == "cvd":
    criterion_rppg = NegativeMaxCrossCorr(180,42)
    criterion_fft = MapPSD("l1")

criterion_fft = criterion_fft.to(device)
criterion_rppg = criterion_rppg.to(device)

if method == "physunet":
    optimizer = op.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=5e-5, weight_decay=0.05)
if method == "bvpnet":
    optimizer = op.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=5e-5, weight_decay=0.05)
if method == "cvd":
    optimizer = op.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=5e-5, weight_decay=0.05)

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


    # defined directory
    logdir = './records/logs/' + method + '__' + train_dir + '__' + args.fold
    model_saved_path = './records/model/' + method + '__' + train_dir + '__' + args.fold + '/'
    #output_saved_path = './records/output/' + method + '__' + train_dir + '__' + args.fold + '/'

    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # loss_train and loss_valid must be in the same graph
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
        losses_train, losses_rppg_train, losses_hr_train, losses_fft_train \
            = train(train_loader, model, criterion_hr, criterion_rppg, criterion_fft, optimizer, epoch)
        losses_valid, losses_rppg_valid, losses_hr_valid, losses_fft_valid \
            = validate(valid_loader, model, epoch)
        # save the model and output
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
