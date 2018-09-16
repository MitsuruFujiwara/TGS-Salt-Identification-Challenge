
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import requests
from skimage.transform import resize

"""
Utilityを置いとくところ
"""

IMG_SIZE_ORI = 101
IMG_SIZE_TARGET = 101
IOU_THRESHOLDS = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

# とりあえずここで定義しときます
NUM_FOLDS = 3

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

# predict both orginal and reflect x
def predict_result(model,x_test,img_size_target):
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(a) for a in model.predict(np.array([np.fliplr(x) for x in x_test])).reshape(-1, img_size_target, img_size_target)])
    return preds_test / 2.0

def rle_encode(im):
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def filter_image(img):
    if img.sum() < 100:
        return np.zeros(img.shape)
    else:
        return img

def iou(img_true, img_pred):
    i = np.sum((img_true*img_pred) >0)
    u = np.sum((img_true + img_pred) >0)
    if u == 0:
        return u
    return i/u

def iou_metric(imgs_true, imgs_pred):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)

    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1
        else:
            scores[i] = (IOU_THRESHOLDS <= iou(imgs_true[i], imgs_pred[i])).mean()

    return scores.mean()

def line_notify(message):
    f = open('../input/line_token.txt')
    token = f.read()
    f.close
    line_notify_token = token
    line_notify_api = 'https://notify-api.line.me/api/notify'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}  # 発行したトークン
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)
