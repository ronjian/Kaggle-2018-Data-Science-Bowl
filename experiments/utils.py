import numpy as np
from io import open as io_open
from re import search, sub

DATA_ROOT = "../data/"
TRAIN_DIR = DATA_ROOT + "stage1_train/"
TEST_DIR = DATA_ROOT + "stage1_test/"
TRAIN_LABEL_CSV = DATA_ROOT + "stage1_train_labels.csv"
SAVE_PATH = "/Users/user/Desktop/git/ronjian.github.io/assets/2018_Data_Science_Bowl_Notes/"


def runlength_to_3D(obj_runlength, H, W):
    """ transform the run-length object data into H*W*C matrix,  
    just like pixels displaying in image.
    The Object is labeled as WHITE, background is marked as Black"""
    obj_1D = np.zeros((H*W*4,)) # initial background as BLACK
    obj_runlength_list = obj_runlength.split(" ")
    for i in range(len(obj_runlength_list)):
        if i % 2 == 0:
            start_point = int(obj_runlength_list[i])
        else:
            run_length = int(obj_runlength_list[i])
            obj_1D[start_point: start_point + run_length] = 255 # WHITE
    obj_3D = np.stack((obj_1D[0:0+H*W ].reshape(W,H).T,
                          obj_1D[1:1+H*W ].reshape(W,H).T,
                          obj_1D[2:2+H*W ].reshape(W,H).T,
                          obj_1D[3:3+H*W ].reshape(W,H).T
                          ), axis = -1)
    return obj_3D

def obj_select(image_id):
    object_list=[]
    with io_open(TRAIN_LABEL_CSV ,'r') as f:
        for line in f:
            if search(image_id , line) is not None:
                object_list.append(sub(image_id + ",", "", line.strip("\n")))
    return object_list