
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import SGD, adam
from keras import backend as K
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from Utils import loadpkl, upsample, downsample, my_iou_metric, my_iou_metric_2, save2pkl, line_notify, predict_result, iou_metric
from Utils import IMG_SIZE_TARGET, NUM_FOLDS
from Preprocessing import get_input_data
from lovasz_losses_tf import keras_lovasz_softmax

"""
Preprocessingで作成したファイルを読み込み、モデルを学習するモジュール。
学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数も定義してください。
"""

warnings.simplefilter(action='ignore', category=FutureWarning)

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

# Build model
# これを参考に修正 https://www.kaggle.com/shaojiaxin/u-net-with-simple-resnet-blocks-v2-new-loss
def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)

    # 12 -> 25
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)

    # 50 -> 101
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)

    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)

    return output_layer

# k-fold用に作っておきます
def kfold_training(train_df, num_folds, stratified = True, debug= False):

    # coverage_class以外のカラム名
    feats = [f for f in train_df.columns if f not in ['coverage_class']]

    # cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)

    X = np.array(train_df.images.tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)
    Y = np.array(train_df.masks.tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)
    cov = train_df.coverage.values
    depth = train_df.z.values

    # out of foldsの結果保存用
    oof_preds = np.zeros((X.shape[0], X.shape[1], X.shape[2]))

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['coverage_class'])):

        # Create train/validation split stratified by salt coverage
        ids_train, ids_valid = train_df.index.values[train_idx], train_df.index.values[valid_idx]
        x_train, y_train = X[train_idx], Y[train_idx]
        x_valid, y_valid = X[valid_idx], Y[valid_idx]
        cov_train, cov_test = train_df.coverage.values[train_idx], train_df.coverage.values[valid_idx]
        depth_train, depth_test = train_df.z.values[train_idx], train_df.z.values[valid_idx]

        # Data augmentation
        x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
        print("train shape: {}, test shape: {}".format(x_train.shape, y_train.shape))

        # model
        input_layer = Input((IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1))
        output_layer = build_model(input_layer, 16, 0.5)
        model_bin = Model(input_layer, output_layer)

        # 最初にlossをbinary_crossentropyにしたモデルを推定します
        if not(os.path.isfile('../output/unet_best_bin'+str(n_fold)+'.model')):
            model_bin.compile(loss="binary_crossentropy", optimizer=adam(lr = 0.01), metrics=[my_iou_metric])

            early_stopping_bin = EarlyStopping(monitor='my_iou_metric',
                                               mode = 'max',
                                               patience=10,
                                               verbose=1)

            model_checkpoint_bin = ModelCheckpoint('../output/unet_best_bin'+str(n_fold)+'.model',
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

            history_bin = model_bin.fit(x_train, y_train,
                                        validation_data=[x_valid, y_valid],
                                        epochs=50,
                                        batch_size=32,
                                        callbacks=[early_stopping_bin, model_checkpoint_bin, reduce_lr],
                                        verbose=1)

            del model_bin, early_stopping_bin, model_checkpoint_bin, history_bin
            gc.collect()

        # binary_crossentropyで推定したモデルをロード
        model = load_model('../output/unet_best_bin'+str(n_fold)+'.model',
                           custom_objects={'my_iou_metric': my_iou_metric})

        # remove layter activation layer and use losvasz loss
        input_x = model.layers[0].input
        output_layer = model.layers[-1].input
        model = Model(input_x, output_layer)

        # lossをLovasz Lossにしたモデルを推定
        model.compile(loss=keras_lovasz_softmax, optimizer=adam(lr = 0.01), metrics=[my_iou_metric_2])

        early_stopping = EarlyStopping(monitor='val_my_iou_metric_2',
                                       mode='max',
                                       patience=20,
                                       verbose=1)

        model_checkpoint = ModelCheckpoint('../output/unet_best'+str(n_fold)+'.model',
                                           monitor='val_my_iou_metric_2',
                                           mode = 'max',
                                           save_best_only=True,
                                           verbose=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2',
                                      mode = 'max',
                                      factor=0.2,
                                      patience=5,
                                      min_lr=0.0001,
                                      verbose=1)

        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid],
                            epochs=50,
                            batch_size=32,
                            callbacks=[early_stopping, model_checkpoint, reduce_lr],
                            verbose=1)

        # out of foldsの推定結果を保存
        oof_preds[valid_idx] = predict_result(model, x_valid, IMG_SIZE_TARGET)

        # save training history
        plt.plot(history.history['my_iou_metric'][1:])
        plt.plot(history.history['val_my_iou_metric'][1:])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','Validation'], loc='upper left')
        plt.savefig('../output/model_loss'+str(n_fold)+'.png')

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
        ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
        ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
        plt.savefig('../output/train_val_loss'+str(n_fold)+'.png')
        plt.close()

        # メモリ節約のための処理
        del ids_train, ids_valid, x_train, y_train, x_valid, y_valid
        del cov_train, cov_test, depth_train, depth_test
        del input_layer, output_layer, model, early_stopping, model_checkpoint, reduce_lr
        del history
        gc.collect()

    # 最終的なIoUスコアを表示
    print('Full IoU score %.6f' % iou_metric(Y, oof_preds))

    # out of foldの推定結果を保存
    save2pkl('../output/oof_preds.pkl', oof_preds)

def main():

    # Loading of training/testing ids and depths
    if os.path.isfile('../output/train_df.pkl'):
        train_df = loadpkl('../output/train_df.pkl')
    else:
        train_df, _ = get_input_data()

    # training
    kfold_training(train_df, NUM_FOLDS, stratified = True, debug= False)

    # 完了後にLINE通知を送信
    line_notify('finished Learning.py')

if __name__ == '__main__':
    main()
