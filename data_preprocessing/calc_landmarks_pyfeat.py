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

#Detects landmarks via Retinaface and PFLD, using the py-feat library

#Models used for face and landmark detection, different ones can be found at https://py-feat.org/content/intro.html
face_model = "retinaface"
landmark_model = "pfld"
detector = Detector(face_model = face_model, landmark_model = landmark_model) #specifies the face and landmark model, there are also emotion and au models integrated in py-feat (loaded but not used)

file = sys.argv[1]

cap = cv2.VideoCapture(file)
frame_width = int(cap.get(3))
frame_heigth = int(cap.get(4))
fps = int(cap.get(5))
fps_down = int(fps)
size = (frame_width, frame_heigth)
h = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #0 to total-1

frames = range(0,total_frames)

start = time.time()
print("Working on clip: "+file)
landmarks_array = np.zeros((int(total_frames),68,2))

for frame_no in tqdm(frames):
    landmarks_index = int(frame_no)
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
        print("Oops, no faces were detected")
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
