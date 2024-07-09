'''
Loads the model and runs inference on all scenarios based on the testing fold. After running for all 5 folds models, the output_signals directory contains the signals that can be used to calculate evaluation metrics
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
from models.model_disentangle import HR_estimator_multi_task_STmap
from models.swin_transformer_unet_skip_expand_decoder_sys_nosq import SwinTransformerSys
from utils_dl import setup_seed, AverageMeter, Acc, NegativeMaxCrossCorr, MapPSD, concatenate_output, set_scenario, split_clips, SNR_loss
from utils_trad import butter_bandpass, calc_hr

def norm(arrr):
    return (arrr-np.min(arrr))/(np.max(arrr)-np.min(arrr))

all_signals_list = []

def evaluate_get_signals(valid_loader, model):
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
    for i, (mstmap, yuv_mstmap, bvpmap, bvp,hr_bvp) in enumerate(valid_loader):
        # measur data loading time
        data_time.update(time.time() - end)

        mstmap = mstmap.to(device=device, dtype=torch.float)
        bvpmap = bvpmap.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)
        yuv_mstmap = yuv_mstmap.to(device=device, dtype=torch.float)
        rgbyuv_mstmap = torch.cat([mstmap,yuv_mstmap],dim=1)


        # out_hr, output, feat = model(mstmap)
        with torch.no_grad():
            if method == "physunet":
                output, out_hr, feat = model(mstmap)
            if method == "bvpnet":
                output, feat  = model(yuv_mstmap)
            if method == "cvd":
                out_hr, output, feat = model(rgbyuv_mstmap)


        for b in range(0,output.size()[0]):
            all_signals_list.append(norm(output[b].detach().cpu().numpy()))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0 or i == len(valid_loader) - 1:
            # print(gt_hr*140+40)
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

if method == "physunet":
    seq_len = 576
if method == "cvd":
    seq_len = 300
if method == "bvpnet":
    seq_len = 256

BATCH_SIZE = 8
NUM_WORKERS = 2*BATCH_SIZE

valid_scen = set_scenario("still+illumination+movement+conceal")
fold = int(foldstr) - 1 # folds go from 1 to 5
total_subject_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030']

divided = ([list(x) for x in mit.divide(5, total_subject_list)])

train_div = list(np.arange(0, 5))
train_div.remove(fold)

train_subj = [*divided[train_div[0]], *divided[train_div[1]], *divided[train_div[2]], *divided[train_div[3]]]
valid_subj = divided[fold]  #validate

valid_set = split_clips(valid_subj, valid_scen, seq_len, seq_len)

id_scen_list = []
for vs in valid_set:
    splittino = vs[0].split("/")
    id_scen_list.append(splittino[-3]+"_"+splittino[-1].split("_")[0])

if method == "physunet" or method == "bvpnet":
    transforms = [T.ToTensor(), T.Resize((64, seq_len))]
if method == "cvd":
    transforms = [T.ToTensor(), T.Resize((320, 320))]

transforms = T.Compose(transforms)
valid_dataset = mst(data=valid_set, stride=seq_len, shuffle=False, transform=transforms, seq_len=seq_len)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, drop_last=False)


if method == "cvd":
    model = HR_estimator_multi_task_STmap(video_length = 300)
if method == "physunet":
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

model.load_state_dict(torch.load(pthfile))

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
        np.save(os.path.join(output_dir,id_scen_list[ind-1]),temp_array)
        temp_list = []
        temp_list.append(all_signals_list[ind])
        prev_id_scen = id_scen
