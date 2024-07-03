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
from feat import Detector
from matplotlib.patches import Rectangle
import more_itertools as mit
import itertools


#Models used for face and landmark detection, different ones can be found at https://py-feat.org/content/intro.html
face_model = "retinaface"
landmark_model = "pfld"
detector = Detector(face_model = face_model, landmark_model = landmark_model) #specifies the face and landmark model, there are also emotion and au models integrated in py-feat (loaded but not used)

downsample = 1  #Factor that downsamples framerate

#folder_path_in = "MMSE_HR/All40_images"
#folder_path_out = "MSTmaps"  #the output mstmaps directory will create the same directory structure as input directory

#Splits the dataset in total_parts, of which only current_part will be processed. Processing is relatively slow, so this way the computation can be split by running the script on different parts of the dataset at the same time on different processes/machines
file = sys.argv[1]

cap = cv2.VideoCapture(file)
frame_width = int(cap.get(3))
frame_heigth = int(cap.get(4))
fps = int(cap.get(5))
fps_down = int(fps/downsample)
size = (frame_width, frame_heigth)
h = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #0 to total-1
#total_frames = 120
if downsample > 1:   #when dowsampling, the total frames need to be divisible by the downsample rate
    if total_frames%downsample!=0:
        total_frames+=-(total_frames%downsample)
frames = range(0,total_frames,downsample)

start = time.time()
print("Working on clip: "+file)
landmarks_array = np.zeros((int(total_frames/downsample),68,2))

for frame_no in tqdm(frames):
    landmarks_index = int(frame_no/downsample)
    start_loop = time.time()

    cap.set(1,frame_no)
    ret, frame = cap.read()
    if frame is not None:   #exception in case frame is none
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prev_rgb_frame = rgb_frame
    else:
        rgb_frame = prev_rgb_frame

    detected_faces = detector.detect_faces(rgb_frame)
    if len(detected_faces[0]) >= 1: #If a face isn´t detected the mstmap row stays 0
        detected_landmarks = detector.detect_landmarks(rgb_frame, detected_faces)
        face = detected_faces[0]  #Assuming only 1 face in video
        landmarks = detected_landmarks[0][0]
        landmarks_array[landmarks_index,:,:] = landmarks

    else:
        landmarks_array[landmarks_index,:,:] = np.zeros((68,2))
        print("sheeeeet")
    #print(MST_map_whole_video[:,idx,:])

end = time.time()
dt = end - start
print("Processed in: ",dt,"s")
cap.release()

lnd_filename = file.replace(".mov","_lnd.npy").replace("video","lnd")

lnd_direc = os.path.join(lnd_filename.split("/")[0],lnd_filename.split("/")[1])
if not os.path.exists(lnd_direc):
   os.makedirs(lnd_direc)

np.save(lnd_filename,landmarks_array)
