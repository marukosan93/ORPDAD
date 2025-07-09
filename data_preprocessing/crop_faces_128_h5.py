# Loading required libraries
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from skimage import io
import sys
import cv2
import os
import time
from tqdm import tqdm
from PIL import Image, ImageDraw
from matplotlib.patches import Rectangle, Polygon
import itertools
from scipy.signal import butter, sosfiltfilt, resample, stft, butter, sosfiltfilt, welch,spectrogram
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
#import exiftool
import json
import h5py

#Crops the face videos to 128x128 so that they are face centered and wraps numpy output using h5

def running_mean(x, N):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    print(dim_len)
    for i in range(0,dim_len):
        if N%2 == 0:
            a, b = i - (N-1)//2, i + (N-1)//2 + 2
        else:
            a, b = i - (N-1)//2, i + (N-1)//2 + 1

        #cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b],axis=0)
    return out

def crop_face(img,lmks):
    m,n,c = img.shape
    ROI_cheek_left1 = np.array([0,1,2,31,41,0])
    ROI_cheek_left2 = np.array([2,3,4,5,48,31,2])
    ROI_cheek_right1 = np.array([16,15,14,35,46,16])
    ROI_cheek_right2 = np.array([14,13,12,11,54,35,14])
    ROI_mouth = [5,6,7,8,9,10,11,54,55,56,57,58,59,48,5]
    ROI_forehead = [17,18,19,20,21,22,23,24,25,26]
    forehead = lmks[ROI_forehead]
    left_eye = np.mean(lmks[36:42],axis=0)
    right_eye = np.mean(lmks[42:48],axis=0)
    eye_distance = np.linalg.norm(left_eye-right_eye)

    tmp = (np.mean(lmks[17:22],axis=0)+ np.mean(lmks[22:27],axis=0))/2 - (left_eye + right_eye)/2;
    tmp = (eye_distance/np.linalg.norm(tmp))*0.6*tmp;

    ROI_forehead=(np.vstack((forehead,forehead[-1].reshape(1,2)+tmp.reshape(1,2),forehead[0].reshape(1,2)+tmp.reshape(1,2),forehead[0].reshape(1,2)))).round(0).astype(int)
    rois = [ROI_forehead,lmks[ROI_cheek_left1],lmks[ROI_cheek_left2],lmks[ROI_cheek_right1],lmks[ROI_cheek_right2],lmks[ROI_mouth]]
    rois = np.concatenate(rois)
    min_y = np.min(rois[:,1])
    min_x = np.min(rois[:,0])
    max_y = np.max(rois[:,1])
    max_x = np.max(rois[:,0])
    rnx = abs(max_x-min_x)
    rny = abs(max_y-min_y)
    c_y = int(round(min_y + rny/2))
    c_x =  int(round(min_x + rnx/2))
    rng = max(rnx,rny)#*1.5
    half = int(round(rng/2))
    lmks[:,1] = lmks[:,1] - (c_y-half)
    lmks[:,0] = lmks[:,0] - (c_x-half)
    return img[c_y-half:c_y+half,c_x-half:c_x+half,:], lmks

#Models used for face and landmark detection, different ones can be found at https://py-feat.org/content/intro.html

file = sys.argv[1]
dir = "./"
fps = 30

input_video = os.path.join(dir,file)

video_block_filename = file.replace(".mov","_block128.h5").replace("video/","blockh5/")
block_direc = os.path.join(video_block_filename.split("/")[0],video_block_filename.split("/")[1])

if not os.path.exists(block_direc):
   os.makedirs(block_direc)

cap = cv2.VideoCapture(input_video)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = float(cap.get(5))
size = (frame_width, frame_height)
h = int(cap.get(cv2.CAP_PROP_FOURCC))

codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #0 to total-1

#total_frames = 128
frames = range(0,total_frames)

landmarks_array = np.load(file.replace(".mov","_lnd.npy").replace("video","lnd"))
landmarks_array = running_mean(landmarks_array,5)

all_frames = []
for frame_no in tqdm(frames):
    cap.set(1,frame_no)
    ret, frame = cap.read()
    if frame is not None:   #exception in case frame is none
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prev_rgb_frame = rgb_frame
    else:
        rgb_frame = prev_rgb_frame

    cropped,lmks =  crop_face(rgb_frame,landmarks_array[frame_no,:,:])
    cropped = cv2.resize(cropped, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    all_frames.append(cropped)

video_block = np.stack(all_frames, axis=0)
with h5py.File(video_block_filename, 'w') as f:
    f['video'] = video_block

#np.save(video_block_filename,video_block)
