
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc

from keras import backend as K
from keras import layers
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add, UpSampling2D
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.optimizers import SGD, adam
from keras.utils.data_utils import get_file
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from Utils import loadpkl, upsample, downsample, my_iou_metric, my_iou_metric_2, save2pkl, line_notify, predict_result, iou_metric
from Utils import IMG_SIZE_TARGET, NUM_FOLDS, WEIGHTS_PATH, WEIGHTS_PATH_NO_TOP
from Preprocessing import get_input_data
from lovasz_losses_tf import keras_lovasz_softmax
from UnetResNet34 import UResNet34

"""
Preprocessingで作成したファイルを読み込み、モデルを学習するモジュール。
学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数も定義してください。
"""

warnings.simplefilter(action='ignore', category=FutureWarning)

###################################################################
# Define New Loss Function
###################################################################
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def bce_lovasz_loss(y_true, y_pred):
    """
    通常のbinary crossentropyとlovansz lossの加重平均をとったloss function。
    以下のpaperを参考に追加。
    http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Rakhlin_Land_Cover_Classification_CVPR_2018_paper.pdf
    """
    # binary_crossentropyの計算用にy_predにsoftmax関数を適用します
    e = K.exp(y_pred - K.max(y_pred, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return 0.5 * binary_crossentropy(y_true, e/s) + 0.5 * keras_lovasz_softmax(y_true, y_pred)

def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) +
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss

###################################################################
# k-fold
###################################################################
def kfold_training(train_df, num_folds, stratified = True, debug= False):

    # coverage_class以外のカラム名
    feats = [f for f in train_df.columns if f not in ['coverage_class']]

    # cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=326)

    X = np.array(train_df.images.map(upsample).tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)
    Y = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)
    cov = train_df.coverage.values
    depth = train_df.z.values

    # out of foldsの結果保存用
    oof_preds = np.zeros((X.shape[0], X.shape[1], X.shape[2]))

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['coverage_class'])):

        # Create train/validation split stratified by salt coverage
        ids_train, ids_valid = train_df.index.values[train_idx], train_df.index.values[valid_idx]
        x_train, y_train = X[train_idx], Y[train_idx]
        x_valid, y_valid = X[valid_idx], Y[valid_idx]

        # Data augmentation
        # 左右の反転
        x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

        # 上下の反転
        x_train = np.append(x_train, [np.flipud(x) for x in x_train], axis=0)
        y_train = np.append(y_train, [np.flipud(x) for x in y_train], axis=0)

        # 画像を回転
#        img_180_x = [np.rot90(x,2) for x in x_train]
#        img_180_y = [np.rot90(x,2) for x in y_train]
#        img_270_x = [np.rot90(x,3) for x in x_train]
#        img_270_y = [np.rot90(x,3) for x in y_train]

#        x_train = np.append(x_train, [np.rot90(x,1) for x in x_train], axis=0)
#        y_train = np.append(y_train, [np.rot90(x,1) for x in y_train], axis=0)

#        x_train = np.append(x_train, img_180_x, axis=0)
#        y_train = np.append(y_train, img_180_y, axis=0)

#        x_train = np.append(x_train, img_270_x, axis=0)
#        y_train = np.append(y_train, img_270_y, axis=0)

        # (128, 128, 3)に変換
        x_train = np.repeat(x_train,3,axis=3)
        x_valid = np.repeat(x_valid,3,axis=3)

        print("train shape: {}, test shape: {}".format(x_train.shape, x_valid.shape))

        if not(os.path.isfile('../output/UnetResNet34_pretrained_bce_dice_'+str(n_fold)+'.model')):

            # model
            model = UResNet34(input_shape=(IMG_SIZE_TARGET,IMG_SIZE_TARGET,3))

            # compile
            model.compile(loss=bce_dice_loss,
#                          optimizer='adam',
                          optimizer=SGD(lr=0.01, momentum=0.9, decay=0.0001),
                          metrics=[my_iou_metric])

            early_stopping = EarlyStopping(monitor='val_my_iou_metric',
                                           mode='max',
                                           patience=20,
                                           verbose=1)

            model_checkpoint = ModelCheckpoint('../output/UnetResNet34_pretrained_bce_dice_'+str(n_fold)+'.model',
                                               monitor='val_my_iou_metric',
                                               mode = 'max',
                                               save_best_only=True,
                                               verbose=1)

            reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric',
                                          mode = 'max',
                                          factor=0.5,
                                          patience=5,
                                          min_lr=0.0001,
                                          verbose=1)

            epochs = 200
            batch_size = 64

            history = model.fit(x_train, y_train,
                                validation_data=[x_valid, y_valid],
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[early_stopping, model_checkpoint, reduce_lr],
                                shuffle=True,
                                verbose=1)

            model = load_model('../output/UnetResNet34_pretrained_bce_dice_'+str(n_fold)+'.model',
                               custom_objects={'my_iou_metric': my_iou_metric,
                                               'bce_dice_loss': bce_dice_loss,
#                                               'bce_lovasz_loss':bce_lovasz_loss
                                               })
        else:
            model = load_model('../output/UnetResNet34_pretrained_bce_dice_'+str(n_fold)+'.model',
                               custom_objects={'my_iou_metric': my_iou_metric,
                                               'bce_dice_loss':bce_dice_loss
                                               })

        """
        input_x = model.layers[0].input
        output_layer = model.layers[-1].input

        model = Model(input_x, output_layer)

        model.compile(loss=keras_lovasz_softmax,
                      optimizer=SGD(lr=0.005, momentum=0.9, decay=0.0001),
                      metrics=[my_iou_metric_2])

        early_stopping = EarlyStopping(monitor='val_my_iou_metric_2',
                                       mode='max',
                                       patience=85,
                                       verbose=1)

        model_checkpoint = ModelCheckpoint('../output/UnetResNet34_pretrained_lovasz_'+str(n_fold)+'.model',
                                           monitor='val_my_iou_metric_2',
                                           mode = 'max',
                                           save_best_only=True,
                                           verbose=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2',
                                      mode = 'max',
                                      factor=0.5,
                                      patience=6,
                                      min_lr=0.000001,
                                      verbose=1)

        epochs = 85
        batch_size = 64

        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping, model_checkpoint, reduce_lr],
                            shuffle=True,
                            verbose=1)
        """

        # out of foldsの推定結果を保存
        oof_preds[valid_idx] = predict_result(model, x_valid, IMG_SIZE_TARGET)

        # foldごとのスコアを送信
        line_notify('fold: %d, train_iou: %.4f val_iou: %.4f'
                    %(n_fold+1, max(history.history['my_iou_metric']), max(history.history['val_my_iou_metric'])))

        # メモリ節約のための処理
        del ids_train, ids_valid, x_train, y_train, x_valid, y_valid
        del model, early_stopping, model_checkpoint, reduce_lr
        del history
        gc.collect()

    # 最終的なIoUスコアを表示
    print('Full IoU score %.6f' % iou_metric(Y, oof_preds))

    # 完了後にLINE通知を送信
    line_notify('Full IoU score %.6f' % iou_metric(Y, oof_preds))

    # out of foldの推定結果を保存
    save2pkl('../output/oof_preds.pkl', oof_preds)

def main():

    # Loading of training/testing ids and depths
    if os.path.isfile('../output/train_df.pkl'):
        train_df = loadpkl('../output/train_df.pkl')
    else:
        train_df, _ = get_input_data()

    # train dataからcoverageが0のものを除外します
#    train_df = train_df[train_df['is_salt']==1]

    # training
    kfold_training(train_df, NUM_FOLDS, stratified = True, debug= False)

if __name__ == '__main__':
    main()
