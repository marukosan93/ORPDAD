import os
import numpy as np
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import cv2
from scipy.signal import butter, sosfiltfilt, resample, stft, butter, sosfiltfilt, welch
import math
from PIL import Image
import torch
from scipy.fft import fft,fftfreq
import torchvision.transforms.functional as transF
import heartpy as hp
from skimage.transform import resize
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
from utils_trad import calc_hr,butter_bandpass

#Dataloader for non-end-to-end methods CVD and PhySU-Net
def RGB2YUV(rgb):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv

class mst(Dataset):
    def __init__(self, data, stride, shuffle=True, transform=None,seq_len=576):
        self.data = data
        self.transform = transform
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        pathname = self.data[idx][0]
        shift = self.data[idx][1]

        mstmap = np.load(pathname)
        fps = 30

        #instead of bvp signal from PPG device, we use a pseudobvp calculated from the ECG signals as it is more accurate
        bvp = np.load(pathname.replace("mstmap", "bvp").replace("maps","bvp"))
        ecg = np.load(pathname.replace("mstmap","ecg").replace("maps","ecg"))
        pseudobvp = np.load(pathname.replace("mstmap","pseudobvp").replace("maps","ecg"))

        mstmap = mstmap[:,shift:int(self.seq_len)+shift,:]
        bvp = bvp[shift:int(self.seq_len)+shift]
        ecg = ecg[shift:int(self.seq_len)+shift]
        pseudobvp = pseudobvp[shift:int(self.seq_len)+shift]

        bvp = butter_bandpass(bvp, 0.5, 3, fps) #low pass filter
        bvp = (bvp-np.min(bvp))/(np.max(bvp)-np.min(bvp))
        ecg = (ecg-np.min(ecg))/(np.max(ecg)-np.min(ecg))
        pseudobvp = (pseudobvp-np.min(pseudobvp))/(np.max(pseudobvp)-np.min(pseudobvp))

        bvpmap = np.stack([pseudobvp]*64,axis=0)
        bvpmap = np.stack([bvpmap]*6,axis=2)

        for idx in range(0,mstmap.shape[0]):
            for c in range(0,3):
                temp = mstmap[idx,:,c]
                mstmap[idx,:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255;

        resized_mstmap = np.zeros((64,self.seq_len,3))

        for i in range(0,self.seq_len):
            row1 = mstmap[:,[i],:]
            row = cv2.resize(row1, dsize=(1,64), interpolation=cv2.INTER_CUBIC)
            resized_mstmap[:,[i],:] = row

        for idx in range(0,resized_mstmap.shape[0]):
            for c in range(0,3):
                temp = resized_mstmap[idx,:,c]
                resized_mstmap[idx,:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255;

        mstmap = resized_mstmap
        yuv_mstmap = RGB2YUV(mstmap)

        stacked_bvpmap = bvpmap
        stacked_bvpmap = ((stacked_bvpmap-np.min(stacked_bvpmap))/(np.max(stacked_bvpmap)-np.min(stacked_bvpmap)))*255

        mstmap1 = mstmap[:,:,0:3].astype(np.uint8())
        yuv_mstmap1 = yuv_mstmap[:,:,0:3].astype(np.uint8())

        bvpmap1 = stacked_bvpmap[:,:,0:3].astype(np.uint8())

        mstmap1 = Image.fromarray(mstmap1)
        yuv_mstmap1 = Image.fromarray(yuv_mstmap1)

        bvpmap1 = Image.fromarray(bvpmap1)

        mstmap1 = self.transform(mstmap1)
        yuv_mstmap1 = self.transform(yuv_mstmap1)
        bvpmap1 = self.transform(bvpmap1)

        hr,_,_,_ = calc_hr(pseudobvp,fps,harmonics_removal=False)

        sample = (mstmap1,yuv_mstmap1,bvpmap1,pseudobvp,hr)
        return sample
