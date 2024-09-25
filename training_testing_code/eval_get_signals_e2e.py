'''
Loads the model and runs inference on all scenarios based on the testing fold. After running for all 5 folds models, the output_signals directory contains the signals that can be used to calculate evaluation metrics
'''

import argparse
import os
import pickle
import time

import more_itertools as mit
import numpy as np
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
from utils_dl import setup_seed, AverageMeter, Acc, NegativeMaxCrossCorr, MapPSD, concatenate_output, set_scenario,split_clips
from utils_trad import butter_bandpass, calc_hr
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


all_signals_list = []

def evaluate_get_signals(valid_loader, model):
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

    end = time.time()
    for i, (input,bvp,hr_bvp) in enumerate(valid_loader):
        # measur data loading time
        data_time.update(time.time() - end)

        input = input.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)
        hr_bvp = hr_bvp.to(device)

        criterion_rppg = nn.MSELoss()
        criterion_rppg2 = NegativeMaxCrossCorr(180,42)

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
                bvp = bvp.view(b,t)
                output = output.view((b,t))
            if method == "physnet":
                output = model(input)
            if method == "physformer":
                output = model(input,2.0)
            if method == "contrastphys":
                model_output = model(input)
                output = model_output[:,-1]

        for b in range(0,output.size()[0]):
            all_signals_list.append((output[b].detach().cpu().numpy()))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0 or i == len(valid_loader) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\n'.format(
                0, i, len(valid_loader), batch_time=batch_time,
                data_time=data_time))

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--loadmodel', type=str, required=True)
args = parser.parse_args()

pthfile = args.loadmodel

print(pthfile)
splitfile = pthfile.split("/")[-2].split("__")
method = splitfile[0]
trainscens = splitfile[1]
foldstr = splitfile[2]

if method == "physnet" or method == "physformer":
    seq_len = 128
    valid_stride = seq_len
if method == "contrastphys":
    valid_stride = 300
    seq_len = 300
if method == "tscan"  or method == "efficientphys" or method == "deepphys":
    frame_depth = 10
    valid_stride = 180#frame_depth
    seq_len = 180#frame_depth#+1

BATCH_SIZE = 1
NUM_WORKERS = 2*BATCH_SIZE

# still = ["S1", "S2", "S3"]
# illumination = ["I1", "I2", "I3", "I4", "I5", "I6"]
# movement = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11"]
# conceal = ["C1", "C2", "C3", "C4", "C5", "C6"]

valid_scen = set_scenario("still+illumination+movement+conceal")
fold = int(foldstr) - 1 # folds go from 1 to 5
total_subject_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030']

divided = ([list(x) for x in mit.divide(5, total_subject_list)])

train_div = list(np.arange(0, 5))
train_div.remove(fold)

train_subj = [*divided[train_div[0]], *divided[train_div[1]], *divided[train_div[2]], *divided[train_div[3]]] #train
valid_subj = divided[fold]  #validate

valid_set = split_clips(valid_subj, valid_scen, seq_len, seq_len)

id_scen_list = []
for vs in valid_set:
    splittino = vs[0].split("/")
    id_scen_list.append(splittino[-3]+"_"+splittino[-1].split("_")[0])

if method == "tscan"  or method == "efficientphys" or method == "deepphys":
    resize_size=72
else:
    resize_size=128

if method == "tscan" or method == "efficientphys" or method == "deepphys":
    valid_dataset = block_deepphys(data=valid_set,stride=valid_stride,shuffle=False, resize_size=resize_size,seq_len=seq_len)
else:
    valid_dataset = block(data=valid_set,stride=valid_stride,shuffle=False, resize_size=resize_size,seq_len=seq_len)

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

model.load_state_dict(torch.load(pthfile))
model.to(device)

evaluate_get_signals(valid_loader, model)
prev_id_scen = id_scen_list[0]
output_dir = os.path.join("output_signals",method,trainscens)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

##Works only if the clips are shorter than video, don't reuse for other datasets without making changes
temp_list = []
for ind, id_scen in enumerate(id_scen_list):
    if id_scen == prev_id_scen and ind != len(id_scen_list)-1:
        temp_list.append(all_signals_list[ind])
        prev_id_scen = id_scen
    else:
        if ind == len(id_scen_list)-1:
            temp_list.append(all_signals_list[ind])
        temp_array = np.concatenate(temp_list,axis=-1)
        temp_array = _detrend(np.cumsum(temp_array), 100)
        np.save(os.path.join(output_dir,id_scen_list[ind-1]),temp_array)
        temp_list = []
        temp_list.append(all_signals_list[ind])
        prev_id_scen = id_scen
