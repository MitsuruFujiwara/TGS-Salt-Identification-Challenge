
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from skimage.transform import resize


"""
Utilityを置いとくところ
"""

IMG_SIZE_ORI = 101
IMG_SIZE_TARGET = 101

# IoUの表示用
def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

# salt coverage計算用
def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i


def save2pkl(path, df):
    f = open(path, 'wb')
    pickle.dump(df, f)
    f.close


def loadpkl(path):
    f = open(path, 'rb')
    out = pickle.load(f)
    return out

# upsamleとdownsample用
def upsample(img):
    if IMG_SIZE_ORI == IMG_SIZE_TARGET:
        return img
    return resize(img, (IMG_SIZE_TARGET, IMG_SIZE_TARGET), mode='constant', preserve_range=True)

def downsample(img):
    if IMG_SIZE_ORI == IMG_SIZE_TARGET:
        return img
    return resize(img, (IMG_SIZE_ORI, IMG_SIZE_ORI), mode='constant', preserve_range=True)
