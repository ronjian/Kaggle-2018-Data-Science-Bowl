import numpy as np
from io import open as io_open
from re import search, sub
from scipy.ndimage import imread
import os
import random

# Constant value
DATA_ROOT = "../data/"
TRAIN_DIR = DATA_ROOT + "stage1_train_images/"
MASK_DIR = DATA_ROOT + "stage1_train_class_256x256/masks/"
TEST_DIR = DATA_ROOT + "stage1_test_images/"
TRAIN_LABEL_CSV = DATA_ROOT + "stage1_train_labels.csv"
TRAIN_IMAGE_IDS = [x.split('.')[0] for x in os.listdir(TRAIN_DIR) if x.endswith(".png")]
random.shuffle(TRAIN_IMAGE_IDS)
TRAIN_IMAGE_CNT = len(TRAIN_IMAGE_IDS)
TEST_IMAGE_IDS = [x.split('.')[0] for x in os.listdir(TEST_DIR) if x.endswith(".png")]
TEST_IMAGE_CNT = len(TEST_IMAGE_IDS)



def runlength_to_3D(obj_runlength, H, W):
    """ transform the run-length object data into H*W*C matrix,  
    just like pixels displaying in image.
    The Object is labeled as WHITE, background is marked as Black"""
    obj_1D = np.zeros((H*W,)) # initial background as BLACK
    obj_runlength_list = obj_runlength.split(" ")
    for i in range(len(obj_runlength_list)):
        if i % 2 == 0:
            start_point = int(obj_runlength_list[i])
        else:
            run_length = int(obj_runlength_list[i])
            obj_1D[start_point: start_point + run_length] = 255 # WHITE
    return obj_1D.reshape(W,H).T

def obj_select(image_id):
    """fetch all object run-length data for one image"""
    object_list=[]
    with io_open(TRAIN_LABEL_CSV ,'r') as f:
        for line in f:
            if search(image_id , line) is not None:
                object_list.append(sub(image_id + ",", "", line.strip("\n")))
    return object_list

def image2ndarry(image_id, folder):
    """generate image's ndarry in shape H*W*C """
    if folder == "train":
        image_path = TRAIN_DIR + image_id + ".png"
    elif folder == "test":
        image_path = TEST_DIR + image_id +  ".png"
    elif folder == "mask":
        image_path = MASK_DIR + image_id +  ".png"
    img_np = imread(image_path) # shape H*W*C
    return img_np


def mask2ndarry(image_id, H, W):
    """form mask's ndarry in shape H*W*C"""
    object_list = obj_select(image_id)
    obj_runlength_concated = " ".join(object_list)
    img_np = runlength_to_3D(obj_runlength_concated, H, W)
    return img_np


def extend_sides(origin):
    """extend the given ndarry in shape H*W*C to 9 times in shape 3H*3W*C 
    The 2D transformation view is as:
    from: |---|---|
          | A | B |
          |---|---|
          | C | D |
          |---|---|
    to:   
          |---|---|---|---|---|---|
          | d | c | c | d | d | c |
          |---|---|---|---|---|---|
          | b | a | a | b | b | a |
          |---|---|---|---|---|---|
          | b | a | A | B | b | a |
          |---|---|---|---|---|---|
          | d | c | C | D | d | c |
          |---|---|---|---|---|---|
          | d | c | c | d | d | c |
          |---|---|---|---|---|---|
          | b | a | a | b | b | a |
          |---|---|---|---|---|---|
    """
    MARGIN = 100
    horizontal = np.fliplr(origin)
    vertical = np.flipud(origin)
    cornor = np.flipud(horizontal)
    row1 = np.concatenate((cornor,vertical,cornor), axis=1)
    row2 = np.concatenate((horizontal,origin,horizontal), axis=1)
    row3 = np.concatenate((cornor,vertical,cornor), axis=1)
    full_size = np.concatenate((row1,row2,row3), axis=0)
    H, W, _ = origin.shape
    H_S = H - MARGIN
    W_S = W - MARGIN
    H_E = 2*H + MARGIN
    W_E = 2*W + MARGIN
    final = full_size[H_S:H_E, W_S:W_E, :]
    return final
    
def iou(gt, pred):
    """IoU between two matrix, if IoU =0 means no intersection."""
    gt_mask = np.squeeze(gt) >0
    pred_mask = np.squeeze(pred) >0
    intersection_cnt = np.sum(gt_mask * pred_mask)
    union_cnt = np.sum(gt_mask) + np.sum(pred_mask) - intersection_cnt
    iou = float(intersection_cnt) / float(union_cnt)
    return iou
