import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from utils_dl import set_scenario
from utils_trad import calc_hr, get_stats,butter_bandpass
import math
from scipy import signal
from scipy.fft import fft, rfft

def spec_hr(sig, fs=30):
    f_min = 0.5
    f_max = 3.0
    # get heart rate by FFT
    # return both heart rate and PSD
    sig = butter_bandpass(sig, 0.5, 3, fs)
    sig = signal.detrend(sig)
    sig = np.pad(sig,128)
    sig = (sig * signal.windows.hann(sig.shape[0]))[128:-128]
    Pxx = np.abs(rfft(sig,int(len(sig)*5*fs)))
    #f, Pxx = welch(sig, 30, nperseg=160,nfft=2048)
    f = np.linspace(0,15,len(Pxx))
    f_hr = f[np.argmax(Pxx)]#*60
    rangino = 2.5
    rangino = rangino/60

    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    Pxx[np.where(f<=f_min)] = 0
    Pxx[np.where(f>=f_max)] = 0

    #Pxx[np.where((f>=f_hr-rangino) & (f<=f_hr+rangino))] = 0
    power_peak = np.sum(Pxx[np.where((f>=f_hr-rangino) & (f<=f_hr+rangino))]) + np.sum(Pxx[np.where((f>=2*f_hr-rangino) & (f<=2*f_hr+rangino))])
    power_rest = np.sum(Pxx[np.where((f>f_min) & (f<f_hr-rangino))]) + np.sum(Pxx[np.where((f>f_hr+rangino) & (f<2*f_hr-rangino))]) + np.sum(Pxx[np.where((f>2*f_hr+rangino) & (f<f_max))])
    snr = 10*math.log10(power_peak/power_rest)
    #print((f[1]-f[0])*60)

    #f, Pxx = welch(sig, 1, nperseg=64,nfft=1024)
    #f = f * fs
    return f,Pxx,snr


def norm(arrr):
    return (arrr-np.min(arrr))/(np.max(arrr)-np.min(arrr))


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--targetsigfolder', type=str, required=True)
parser.add_argument('-s', '--eval_scenarios', type=str, required=True)
args = parser.parse_args()

signal_folder = args.targetsigfolder
eval_scenarios = args.eval_scenarios
fps = 30
eval_scen = set_scenario(eval_scenarios)
files = os.listdir(signal_folder)
#if len(files)/26 < 20: Do this when doing all five folds stats
#    print("folds missing")

files_to_eval = []
for file in files:
    if file.split("_")[1][:-4] in eval_scen:
        files_to_eval.append(file)

hr_pbvp_list = []
hr_device_list = []
hr_bvp_list = []
hr_output_list = []
snr_output_list = []
for file in files_to_eval:
    output = np.load(os.path.join(signal_folder,file))
    pbvp = np.load(os.path.join("subjects",file.split("_")[0],"ecg",file.split("_")[1].replace(".npy","_pseudobvp.npy")))
    bvp = np.load(os.path.join("subjects",file.split("_")[0],"bvp",file.split("_")[1].replace(".npy","_bvp.npy")))
    hr_device = np.load(os.path.join("subjects",file.split("_")[0],"bvp",file.split("_")[1].replace(".npy","_hr.npy")))

    mid_index = int(round(output.shape[-1]/2))
    mid_pbvp = int(round(len(pbvp)/2))
    mid_bvp = int(round(len(bvp)/2))
    mid_hr = int(round(len(hr_device)/2))
    pbvp = pbvp[mid_pbvp-mid_index:mid_pbvp+mid_index]
    bvp = bvp[mid_bvp-mid_index:mid_bvp+mid_index]
    clip_len = 30 * fps
    if mid_index < clip_len:
        clip_len = mid_index
    for clip_num in range(0,2):

        if len(output.shape) > 1:
            output_clip = output[:,:,mid_index+(clip_num-1)*clip_len:mid_index+(clip_num)*clip_len]
            #output_clip = np.mean(output_clip,axis=(0,1)) # might need to change this to bvpnets method later
            list_outputs_map = []
            list_outputs_snr = []
            for roi in range(0,output_clip.shape[1]):
                for ch in range(0,output_clip.shape[0]):
                    hr_output_clip, pxx,f,filtsig = calc_hr(output_clip[ch,roi,:],harmonics_removal=True)
                    f_out, pxx_out, snr_out = spec_hr(output_clip[ch,roi,:])

                    list_outputs_map.append(hr_output_clip)
                    #list_outputs_snr.append(snr_out)

            list_outputs_map_arr = np.array(list_outputs_map)
            list_outputs_map.sort()
            sort_ind = np.argsort(list_outputs_map_arr)
            cutoff = int(round(len(list_outputs_map)*0.2))
            hr_output_clip = np.mean(list_outputs_map[cutoff:-cutoff])
            #list_outputs_snr = [list_outputs_snr[kk] for kk in sort_ind]
            #snr_output_clip =  np.mean(list_outputs_snr[cutoff:-cutoff])
            snr_output_clip = 0
        else:
            output_clip = output[mid_index+(clip_num-1)*clip_len:mid_index+(clip_num)*clip_len]

            hr_output_clip, pxx_out,f_out,filtsig_out = calc_hr(output_clip,harmonics_removal=True)
            f_out, pxx_out, snr_output_clip = spec_hr(output_clip)

        pbvp_clip = pbvp[mid_index+(clip_num-1)*clip_len:mid_index+(clip_num)*clip_len]
        bvp_clip = bvp[mid_index+(clip_num-1)*clip_len:mid_index+(clip_num)*clip_len]

        hr_pbvp_clip, pxx_pbvp,f_pbvp, filtsig_pbvp = calc_hr(pbvp_clip,harmonics_removal=True)
        hr_bvp_clip, pxx_bvp,f_bvp, filtsig_bvp = calc_hr(bvp_clip,harmonics_removal=True)
        hr_device_clip = np.mean(hr_device[mid_hr+30*(clip_num-1):mid_hr+30*clip_num])

        #plt.plot(norm(output_clip))
        #plt.plot(norm(pbvp_clip))
        #plt.show()
        #exit()
        error_pbvp = abs(hr_pbvp_clip - hr_output_clip)
        if error_pbvp >5:
            print(file,"----",error_pbvp)
        error = abs(hr_pbvp_clip - hr_output_clip)
        error_bvp = abs(hr_bvp_clip - hr_output_clip)
        #print("Err_pbvp: ",error)
        print("Err_bvp: ",error_bvp)
        if error_pbvp > 5:
            print(file," ",clip_num)
            print(error_pbvp)
            print(hr_pbvp_clip)
            if len(output.shape) > 1:
                fig,ax = plt.subplots(2,1)
                ax[0].plot(output_clip[1,10,:])
                #ax[1].plot(pbvp_clip[:])
                ax[1].plot(pbvp_clip[:])
                #ax[0].plot(norm(filtsig_out))
                #ax[0].plot(norm(filtsig_pbvp))
                #ax[1].plot(f_out,norm(pxx_out))
                #ax[1].plot(f_pbvp,norm(pxx_pbvp))
                plt.show()
            else:
                fig,ax = plt.subplots(2,1)
                ax[0].plot(norm(filtsig_out))
                ax[0].plot(norm(filtsig_pbvp))
                ax[1].plot(f_out,norm(pxx_out))
                ax[1].plot(f_pbvp,norm(pxx_pbvp))
                plt.show()

        hr_pbvp_list.append(hr_pbvp_clip)
        hr_bvp_list.append(hr_bvp_clip)
        hr_output_list.append(hr_output_clip)
        hr_device_list.append(hr_device_clip)
        snr_output_list.append(snr_output_clip)

mae,rmse,std,r_score = get_stats(hr_pbvp_list,hr_output_list)
snr = np.mean(snr_output_list)
print('Model :'+args.targetsigfolder+'  -  EvalScens :'+args.eval_scenarios+'  -  MAE: '+str(mae)+'  -  RMSE: '+str(rmse)+'  -  R: '+str(r_score)+'  -  SNR: '+str(snr)+'\n')
with open("stats.txt", "a") as file_object:
    # Append 'hello' at the end of file
    #file_object.write('Model :'+name_of_run+'  -  MAE: '+str(acc.mae)+'  -  RMSE: '+str(acc.rmse)+'  -  STD: '+str(acc.std)+'\n')
    file_object.write('Model :'+args.targetsigfolder+'  -  EvalScens :'+args.eval_scenarios+'  -  MAE: '+str(mae)+'  -  RMSE: '+str(rmse)+'  -  R: '+str(r_score)+'  -  SNR: '+str(snr)+'\n')
"""print('Model :'+args.targetsigfolder+'  -  EvalScens :'+args.eval_scenarios+'  -  MAE: '+str(mae)+'  -  RMSE: '+str(rmse)+'  -  R: '+str(r_score)+'  -  STD: '+str(std)+'\n')
with open("stats.txt", "a") as file_object:
    # Append 'hello' at the end of file
    #file_object.write('Model :'+name_of_run+'  -  MAE: '+str(acc.mae)+'  -  RMSE: '+str(acc.rmse)+'  -  STD: '+str(acc.std)+'\n')
    file_object.write('Model :'+args.targetsigfolder+'  -  EvalScens :'+args.eval_scenarios+'  -  MAE: '+str(mae)+'  -  RMSE: '+str(rmse)+'  -  R: '+str(r_score)+'  -  STD: '+str(std)+'\n')"""
