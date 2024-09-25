import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
import more_itertools as mit

def list_files(dir,extension):
    r = []
    for file in os.listdir(dir):
        dirpath = os.path.join(dir, file)
        if file[-len(extension):] == extension:
            r.append(dirpath)
    return r


def VIPL_split(whatfold):
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.
    list_subj = []
    for s in list(np.arange(1,108,1).astype(str)):
        s = "/p"+s+"_"
        list_subj.append(s)

    h5_dir = './Datasets/h5_vipl/'
    train_list = []
    val_list = []

    if whatfold != "whole":
        divided = ([list(x) for x in mit.divide(5,list_subj)])
        print("FOLD",whatfold)
        fold = whatfold - 1
        train_div = list(np.arange(0,5))
        train_div.remove(fold)

        train_subj = [*divided[train_div[0]], *divided[train_div[1]],*divided[train_div[2]],*divided[train_div[3]]] #train
        valid_subj = divided[fold] #validate
    else:
        train_subj = list_subj
        valid_subj = []

    all_files = list_files(h5_dir,".h5")   #gets all filepaths that contain extension
    all_files.sort()

    for file in all_files:
        for s in train_subj:
            if s in file:
                train = True
        for s in valid_subj:
            if s in file:
                train = False
        if train:
            train_list.append(file)
        else:
            val_list.append(file)
    return train_list, val_list


def OBF_split(whatfold):
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.
    list_subj = []
    for s in list(np.arange(1,101,1).astype(str)):
        if len(s) == 1:
            s = "00"+s
        if len(s) == 2:
            s = "0"+s
        list_subj.append(s)


    h5_dir = './Datasets/h5_obf/'
    train_list = []
    val_list = []

    if whatfold != "whole":
        divided = ([list(x) for x in mit.divide(10,list_subj)])
        print("FOLD",whatfold)
        fold = whatfold - 1
        train_div = list(np.arange(0,10))
        train_div.remove(fold)

        train_subj = [*divided[train_div[0]], *divided[train_div[1]], *divided[train_div[2]],*divided[train_div[3]],*divided[train_div[4]],*divided[train_div[5]],*divided[train_div[6]],*divided[train_div[7]],*divided[train_div[8]]] #train
        valid_subj = divided[fold] #validate
    else:
        train_subj = list_subj
        valid_subj = []

    all_files = list_files(h5_dir,".h5")   #gets all filepaths that contain extension
    all_files.sort()

    for file in all_files:
        for s in train_subj:
            if s in file:
                train = True
        for s in valid_subj:
            if s in file:
                train = False
        if train:
            train_list.append(file)
        else:
            val_list.append(file)
    return train_list, val_list


def MMSE_split(whatfold):
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.
    list_subj = []
    for s in range(5,28):
        list_subj.append("F"+str(s).zfill(3))
    for s in range(1,18):
        list_subj.append("M"+str(s).zfill(3))

    list_subj.remove("F020")

    h5_dir = './Datasets/h5_mmse/'
    train_list = []
    val_list = []

    if whatfold != "whole":

        divided = ([list(x) for x in mit.divide(3,list_subj)])
        print("FOLD",whatfold)
        fold = whatfold - 1
        train_div = list(np.arange(0,3))
        train_div.remove(fold)

        train_subj = [*divided[train_div[0]], *divided[train_div[1]]] #train
        valid_subj = divided[fold] #validate
    else:
        train_subj = list_subj
        valid_subj = []

    all_files = list_files(h5_dir,".h5")   #gets all filepaths that contain extension
    all_files.sort()

    for file in all_files:
        for s in train_subj:
            if s in file:
                train = True
        for s in valid_subj:
            if s in file:
                train = False
        if train:
            train_list.append(file)
        else:
            val_list.append(file)
    return train_list, val_list

def PURE_split():
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.
    list_subj = []
    for s in range(1,11):
        list_subj.append(str(s).zfill(2))

    h5_dir = './Datasets/h5_pure/'
    train_list = []
    val_list = []


    train_subj = ['01-', '02-', '03-', '04-', '05-', '07-']
    valid_subj = ['06-', '08-', '09-', '10-']

    all_files = list_files(h5_dir,".h5")   #gets all filepaths that contain extension
    all_files.sort()

    for file in all_files:
        for s in train_subj:
            if s in file:
                train = True
        for s in valid_subj:
            if s in file:
                train = False
        if train:
            train_list.append(file)
        else:
            val_list.append(file)
    return train_list, val_list

class H5Dataset(Dataset):

    def __init__(self, train_list, T):
        self.train_list = train_list # list of .h5 file paths for training
        self.T = T # video clip length

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = f['imgs'].shape[0]
            if img_length-self.T == 0:
                idx_start = 0
            else:
                idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        return img_seq
