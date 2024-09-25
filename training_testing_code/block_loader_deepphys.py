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

# For  DeepPhys and TS-CAN

def standardized_data(data):
    """Z-score standardization for video data."""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data

def diff_normalize_label(label):
    """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
    diff_label = np.diff(label, axis=0)
    diffnormalized_label = diff_label / np.std(diff_label)
    diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
    diffnormalized_label[np.isnan(diffnormalized_label)] = 0
    return diffnormalized_label

def diff_normalize_data(data):
    """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data


class block_deepphys(Dataset):
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

        ecg = (ecg-np.min(ecg))/(np.max(ecg)-np.min(ecg))
        pseudobvp = (pseudobvp-np.min(pseudobvp))/(np.max(pseudobvp)-np.min(pseudobvp))

        video_tensor = torch.FloatTensor(video_block/255).permute(3,0,1,2)
        if self.resize_size != 128:
            video_tensor = transF.resize(video_tensor,self.resize_size)

        numpy_ressampled = video_tensor.permute(1,0,2,3).numpy()

        std_data = standardized_data(numpy_ressampled)
        diffnorm_data = diff_normalize_data(numpy_ressampled)
        pseudobvp = diff_normalize_label(pseudobvp)

        video_block = np.concatenate([diffnorm_data,std_data],axis=1)
        video_tensor = torch.FloatTensor(video_block)#.permute(3,0,1,2)

        if video_tensor.size()[1] > 21: #should be 11 in case of mtsscan
            hr,_,_,_ = calc_hr(pseudobvp,fps,harmonics_removal=False)
        else:
            hr = 0 # for mtsscan it's not needed and gives error since T is too short
        sample = (video_tensor,pseudobvp,hr)
        return sample
