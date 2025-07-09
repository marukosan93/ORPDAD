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


# Python implementation of MSTmap from Niu, X. et al. "Video-based remote physiological measurement via cross-verified feature disentangling." Computer Vision–ECCV 2020

#Outputs permutations of every single of the ROI combinarions, e.g. 000001,000010,..,111111
def comb():
    X = [0,1]
    result = np.zeros((63,6))
    number=-1
    for combination in itertools.product(X,X,X,X,X,X):
        number+=1
        if number > 0:
            result[number-1,:] = np.array(combination)
    return result.astype(int)

#Computes N-point running mean
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

#transforms polygon to binary mask
def poly2mask(polyarray,m,n):
    img = Image.new('L', (n, m), 0)
    ImageDraw.Draw(img).polygon(polyarray.flatten().tolist(), outline=1, fill=1)
    mask = np.array(img)
    return mask

def get_combined_signal_map(SignalMap,ROInum,All_idx):
    smap = SignalMap.copy()
    rnum = ROInum.copy()
    SignalMapOut = np.zeros((len(All_idx),1,3))
    for index in range(0,len(All_idx)):
        tmp_idx = np.where(All_idx[index]==1)
        tmp_signal = smap[tmp_idx,:]
        tmp_ROI = rnum[tmp_idx]
        tmp_ROI = tmp_ROI/np.sum(tmp_ROI)
        tmp_ROI = matlib.repmat(tmp_ROI,1,3)
        SignalMapOut[index,:,:] = np.sum(tmp_signal*tmp_ROI,axis=1)
    return SignalMapOut

def get_ROI_signal(img,mask):
    m,n,c = img.shape
    signal = np.zeros((1,1,c))
    signal2 = np.zeros((1,1,c))
    for i in range(0,c):
        tmp = img[:,:,i]
        signal[0,0,i] = np.sum(tmp*mask)/np.sum(mask)
    return signal

#reduces ROI size, so that there are less pixels bordering with outside of the face
def shrinkroi(landsss,scala):
    shift = np.min(landsss,axis=0)+(np.max(landsss,axis=0)-np.min(landsss,axis=0))/2
    landsss = landsss - shift
    landsss = landsss *scala
    landsss = landsss + shift
    return landsss

def generate_signal_map(img,lmks,signal_map,mst_signal_map,background_map,idx,dir,file):
    file = file.replace("video",dir)
    original = img.copy() #DEBUG
    m,n,c = img.shape
    image = img[:,:,:]
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
    scala = 0.80
    lmks[ROI_cheek_left1] = shrinkroi(lmks[ROI_cheek_left1],scala)
    lmks[ROI_cheek_left2] = shrinkroi(lmks[ROI_cheek_left2],scala)
    lmks[ROI_cheek_right1] = shrinkroi(lmks[ROI_cheek_right1],scala)
    lmks[ROI_cheek_right2] = shrinkroi(lmks[ROI_cheek_right2],scala)
    lmks[ROI_mouth] = shrinkroi(lmks[ROI_mouth],scala)
    ROI_forehead = shrinkroi(ROI_forehead,scala)

    mask_ROI_cheek_left1= poly2mask(lmks[ROI_cheek_left1],m,n);
    mask_ROI_cheek_left2 = poly2mask(lmks[ROI_cheek_left2],m,n);
    mask_ROI_cheek_right1 = poly2mask(lmks[ROI_cheek_right1],m,n);
    mask_ROI_cheek_right2 = poly2mask(lmks[ROI_cheek_right2],m,n);
    mask_ROI_mouth  = poly2mask(lmks[ROI_mouth],m,n);
    mask_ROI_forehead = poly2mask(ROI_forehead,m,n);

    masks = [mask_ROI_cheek_left1,mask_ROI_cheek_left2,mask_ROI_cheek_right1,mask_ROI_cheek_right2,mask_ROI_mouth,mask_ROI_forehead]

    b_side = 120

    b_x = 1600
    b_y = 200

    ROIs = ["cheek_left1","cheek_left2","cheek_right1","cheek_right2","mouth","forehead"]

    if idx == 120:
        plt.gca().add_patch(Rectangle((b_x,b_y),b_side,b_side,edgecolor='none',facecolor='red',alpha=0.3))

        plt.imshow(image)
        plt.gca().add_patch(Polygon(lmks[ROI_cheek_left1],edgecolor='none',facecolor='pink',alpha=0.3))
        plt.gca().add_patch(Polygon(lmks[ROI_cheek_left2],edgecolor='none',facecolor='cyan',alpha=0.3))
        plt.gca().add_patch(Polygon(lmks[ROI_cheek_right1],edgecolor='none',facecolor='green',alpha=0.3))
        plt.gca().add_patch(Polygon(lmks[ROI_cheek_right2],edgecolor='none',facecolor='yellow',alpha=0.3))
        plt.gca().add_patch(Polygon(lmks[ROI_mouth],edgecolor='none',facecolor='orange',alpha=0.3))
        plt.gca().add_patch(Polygon(ROI_forehead,edgecolor='none',facecolor='brown',alpha=0.3))


        plt.savefig(os.path.join(file[:-4]+'_rois.png'))
        plt.close()

        #NOTE WRITE for all 3 channels
        fig, ax = plt.subplots(3,1,figsize=(4, 12))
        for ch in range(0,3):
            curr = image[:,:,ch]
            counts, bins = np.histogram(curr, range(257))
            contrast_michelson = (np.max(curr)-np.min(curr))/ (np.max(curr)+np.min(curr))
            contrast_rms = round(np.std(curr),2)
            contrast_michelson = round(contrast_michelson,2)
            intensity_range = np.max(curr)-np.min(curr)
            # plot histogram centered on values 0..255
            ax[ch].bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
            ax[ch].set_xlim([-0.5, 255.5])
            ax[ch].set_title("Range: "+str(intensity_range)+"   |   Crms: "+str(contrast_rms)+"   |   C_mch: "+str(contrast_michelson))

        fig.suptitle("Whole image:"+dir+" - "+file)
        plt.savefig(os.path.join(file[:-4]+'_image_hist.png'))
        plt.close()

        total_mask = np.zeros_like(image[:,:,0])
        for i in range(0,6):
            total_mask = total_mask + masks[i]

        fig, ax = plt.subplots(3,1,figsize=(4, 12))
        for ch in range(0,3):
            curr = image[:,:,ch]
            curr = curr[np.where(total_mask>0)]
            counts, bins = np.histogram(curr, range(257))
            contrast_michelson = (np.max(curr)-np.min(curr))/ (np.max(curr)+np.min(curr))
            contrast_rms = round(np.std(curr),2)
            contrast_michelson = round(contrast_michelson,2)
            intensity_range = np.max(curr)-np.min(curr)
            # plot histogram centered on values 0..255
            ax[ch].bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
            ax[ch].set_xlim([-0.5, 255.5])
            ax[ch].set_title("Range: "+str(intensity_range)+"   |   Crms: "+str(contrast_rms)+"   |   C_mch: "+str(contrast_michelson))
        fig.suptitle("ROI:"+dir+" - "+file)
        plt.savefig(os.path.join(file[:-4]+'_roi_hist.png'))
        plt.close()

    Signal_tmp = np.zeros((6,3))
    ROI_num = np.zeros((6,1))
    Background_tmp = np.sum(image[b_y:b_y+b_side,b_x:b_x+b_side,:],axis=(0,1))/(b_side*b_side)

    #Get ROI calculated
    Signal_tmp[0,:] = get_ROI_signal(img,mask_ROI_cheek_left1)
    Signal_tmp[1,:] = get_ROI_signal(img,mask_ROI_cheek_left2)
    Signal_tmp[2,:] = get_ROI_signal(img,mask_ROI_cheek_right1)
    Signal_tmp[3,:] = get_ROI_signal(img,mask_ROI_cheek_right2)
    Signal_tmp[4,:] = get_ROI_signal(img,mask_ROI_mouth)
    Signal_tmp[5,:] = get_ROI_signal(img,mask_ROI_forehead)

    #Get ROI pixel
    ROI_num[0] = np.sum(mask_ROI_cheek_left1)
    ROI_num[1] = np.sum(mask_ROI_cheek_left2)
    ROI_num[2] = np.sum(mask_ROI_cheek_right1)
    ROI_num[3] = np.sum(mask_ROI_cheek_right2)
    ROI_num[4] = np.sum(mask_ROI_mouth)
    ROI_num[5] = np.sum(mask_ROI_forehead)

    All_idx1 = np.array([[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 1, 0],[0, 0, 0, 1, 0, 0],[0, 0, 1, 0, 0, 0],[0, 1, 0, 0, 0, 0],[1, 0, 0, 0, 0, 0],[1,1,1,1,1,1]])
    All_idx2 = comb()


    signal_map[:,idx,:] = get_combined_signal_map(Signal_tmp,ROI_num,All_idx1).squeeze()
    mst_signal_map[:,idx,:] = get_combined_signal_map(Signal_tmp,ROI_num,All_idx2).squeeze()

    background_map[idx] = Background_tmp
    return signal_map,mst_signal_map,background_map

def list_dirs(dir,extension):
    r = []
    if extension == ".MOV":
        for root, dirs, files in os.walk(dir):
            for dir in dirs:
                dirpath = os.path.join(root, dir)
                for file in os.listdir(dirpath):
                    if file[-len(extension):] == extension:
                        r.append((dirpath,file))
    return r

#Models used for face and landmark detection, different ones can be found at https://py-feat.org/content/intro.html

file = sys.argv[1]
dir = "./"
fps = 30
outdir = "output"

input_video = os.path.join(dir,file)

stmap_filename = file.replace(".mov","_stmap.npy").replace("video","maps")
mstmap_filename = file.replace(".mov","_mstmap.npy").replace("video","maps")
bgmap_filename = file.replace(".mov","_bgmap.npy").replace("video","maps")

map_direc = os.path.join(stmap_filename.split("/")[0],stmap_filename.split("/")[1])
output_direc = map_direc.replace("maps",outdir)

if not os.path.exists(map_direc):
   os.makedirs(map_direc)

if not os.path.exists(output_direc):
   os.makedirs(output_direc)

cap = cv2.VideoCapture(input_video)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = float(cap.get(5))
size = (frame_width, frame_height)
h = int(cap.get(cv2.CAP_PROP_FOURCC))

codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #0 to total-1
frames = range(0,total_frames)

#files = [input_video]

ST_map = np.zeros((7,total_frames,3))
MST_map = np.zeros((63,total_frames,3))
BG_signal = np.zeros((total_frames,3))
landmarks_array = np.load(file.replace(".mov","_lnd.npy").replace("video","lnd"))

landmarks_array = running_mean(landmarks_array,5)

for frame_no in tqdm(frames):
    cap.set(1,frame_no)
    ret, frame = cap.read()
    if frame is not None:   #exception in case frame is none
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prev_rgb_frame = rgb_frame
    else:
        rgb_frame = prev_rgb_frame

    if np.sum(landmarks_array[frame_no,:,:]) > 0:
        ST_map,MST_map, BG_signal = generate_signal_map(rgb_frame,landmarks_array[frame_no,:,:],ST_map,MST_map,BG_signal,frame_no,outdir,file)

for idx in range(0,MST_map.shape[0]):
    for c in range(0,MST_map.shape[-1]):
        temp = MST_map[idx,:,c]
        MST_map[idx,:,c] = (temp - np.min(temp))/(np.max(temp) - np.min(temp))*255



np.save(stmap_filename,ST_map)
np.save(bgmap_filename,BG_signal)
np.save(mstmap_filename,MST_map)
