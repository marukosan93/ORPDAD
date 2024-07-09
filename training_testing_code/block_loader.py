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
from utils_trad import calc_hr, butter_bandpass
import h5py

# For all end-to-end methods except DeepPhys and TS-CAN

class block(Dataset):
    def __init__(self, data,stride,shuffle=True, resize_size=128,seq_len=576):
        self.data = data
        self.resize_size = resize_size
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        pathname = self.data[idx][0]
        shift = self.data[idx][1]

        block_pathname = pathname.replace("mstmap","block128").replace("maps","blockh5").replace(".npy",".h5")

        video_block = h5py.File(block_pathname, 'r')
        fps = 30

        #instead of bvp signal from PPG device, we use a pseudobvp calculated from the ECG signals as it is more accurate
        bvp = np.load(pathname.replace("mstmap","bvp").replace("maps","bvp"))
        ecg = np.load(pathname.replace("mstmap","ecg").replace("maps","ecg"))
        pseudobvp = np.load(pathname.replace("mstmap","pseudobvp").replace("maps","ecg"))

        video_block = video_block["video"][shift:int(self.seq_len)+shift,:,:,:]
        bvp = bvp[shift:int(self.seq_len)+shift]
        ecg = ecg[shift:int(self.seq_len)+shift]
        pseudobvp = pseudobvp[shift:int(self.seq_len)+shift]

        bvp = butter_bandpass(bvp, 0.5, 3, fps) #low pass filter to remove DC component (introduced by normalisation)
        bvp = (bvp-np.min(bvp))/(np.max(bvp)-np.min(bvp))

        ecg = (ecg-np.min(ecg))/(np.max(ecg)-np.min(ecg))
        pseudobvp = (pseudobvp-np.min(pseudobvp))/(np.max(pseudobvp)-np.min(pseudobvp))

        video_tensor = torch.FloatTensor(video_block/255).permute(3,0,1,2)
        if self.resize_size != 128:
            video_tensor = transF.resize(video_tensor,self.resize_size)
        if video_tensor.size()[1] > 33:
            hr,_,_,_ = calc_hr(pseudobvp,fps,harmonics_removal=False)
        else:
            hr = 0
        sample = (video_tensor,pseudobvp,hr)
        return sample
