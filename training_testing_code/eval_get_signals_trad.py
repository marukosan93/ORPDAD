'''
Runs hand-crafted methods on all scenarios. The output_signals directory contains the signals that can be used to calculate evaluation metrics.
'''

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.signal import butter, filtfilt, welch,resample
from utils_trad import GREEN_rppg, CHROM_rppg, POS_rppg, PBV_rppg, LGI_rppg, PCA_rppg, ICA_rppg, calc_hr, butter_bandpass,get_stats,norm
import heartpy as hp



method = sys.argv[1]

print("***************************")
print(method)
print("***************************")


still = ["S1","S2","S3"]
illumination = ["I1","I2","I3","I4","I5","I6"]
movement = ["M1","M2","M3","M4","M5","M6","M7","M8","M9","M10","M11"]
conceal = ["C1","C2","C3","C4","C5","C6"]


save_dir = "./output_signals/"+method+"/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

subjects_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030']
for subject_name in subjects_list:
    for scens in [still,illumination,movement,conceal]:
        hrs_green = []
        hrs_chrom = []
        hrs_pos = []
        hrs_pbv = []
        hrs_lgi = []
        hrs_pca = []
        #hrs_ica = []
        hrs_bvp = []
        hrs_pbvp = []
        hrs_ecg = []

        for ind, scen in enumerate(scens):
            fps = 30
            file = os.path.join("subjects",subject_name,"maps",scen+"_stmap.npy")
            stmap = np.load(file)
            bgsig = np.load(file.replace("stmap","bgmap"))
            bvp = np.load(file.replace("stmap","bvp").replace("maps","bvp"))
            pbvp = np.load(file.replace("stmap","pseudobvp").replace("maps","ecg"))
            ecg = np.load(file.replace("stmap","ecg").replace("maps","ecg"))

            sig = stmap[-1,:,:] #take only last features (signal from whole face)

            if method == "green":
                out_sig = GREEN_rppg(sig)
            if method == "chrom":
                out_sig = CHROM_rppg(sig)
            if method == "pos":
                out_sig = POS_rppg(sig)
            if method == "pbv":
                out_sig = PBV_rppg(sig)
            if method == "lgi":
                out_sig = LGI_rppg(sig)
            if method == "pca":
                out_sig = PCA_rppg(sig)
            if method == "ica":
                out_sig = ICA_rppg(sig)

            out_name = subject_name+"_"+scen+".npy"
            np.save(os.path.join(save_dir,out_name),out_sig)
