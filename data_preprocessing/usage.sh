#!/bin/bash
python calc_landmarks_pyfeat.py 001/video/S1.mov &&   #calculate landmarks an save in ./001/lnd/S1_lnd.npy, skip if landmarks already exist
python mstmap_hist_roi.py 001/video/S1.mov && #calculate mstmaps an save in ./001/maps/S1_mstmap.npy 
python crop_faces_128.py 001/video/S1.mov #crop video to 128x128 face centered an save in ./001/maps/S1_mstmap.npy 
