
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import time
from tqdm import tqdm

from keras.models import Model
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD
from keras import backend as K
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from keras.models import load_model
from keras.preprocessing.image import load_img

from Utils import predict_result, loadpkl, my_iou_metric, rle_encode, filter_image, iou_metric

from Utils import loadpkl, upsample, downsample, my_iou_metric, save2pkl, line_notify, predict_result, iou_metric
from Preprocessing import get_input_data

"""
Preprocessingで作成したファイルを読み込み、モデルを学習するモジュール。
学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数も定義してください。
"""

warnings.simplefilter(action='ignore', category=FutureWarning)

ACTIVATION = "relu"

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation(ACTIVATION)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = Activation(ACTIVATION)(blockInput)
    x = BatchNormalization()(x)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x

# Build model
def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16)
    convm = Activation(ACTIVATION)(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = Activation(ACTIVATION)(uconv4)

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = Activation(ACTIVATION)(uconv2)

    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = Activation(ACTIVATION)(uconv1)

    uconv1 = Dropout(DropoutRatio/2)(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)

    return output_layer

"""
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
        output_layer = build_model(input_layer, 16,0.5)

        model = Model(input_layer, output_layer)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[my_iou_metric])

        early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max', patience=20, verbose=1)
        model_checkpoint = ModelCheckpoint('../output/unet_best'+str(n_fold)+'.model',monitor='val_my_iou_metric', mode = 'max', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.2, patience=5, min_lr=0.00001, verbose=1)

        epochs = 200
        batch_size = 32

        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size,
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
"""


def prediction(train_df, test_df, name):
    # Create train/validation split stratified by salt coverage
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid,\
    cov_train, cov_test, depth_train, depth_test = train_test_split(train_df.index.values,
                                                                    np.array(train_df.images.tolist()).reshape(-1, img_size_target, img_size_target, 1),
                                                                    np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1),
                                                                    train_df.coverage.values,
                                                                    train_df.z.values,
                                                                    test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)

    #Data augmentation
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
    print("train shape: {}, test shape: {}".format(x_train.shape, y_train.shape))

    # model
    input_layer = Input((img_size_target, img_size_target, 1))
    output_layer = build_model(input_layer, 16,0.5)

    model = Model(input_layer, output_layer)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[my_iou_metric])

    early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint("../output/" + name + ".model",monitor='val_my_iou_metric',
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.2, patience=5, min_lr=0.00001, verbose=1)
    #reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, verbose=1)

    epochs = 200
    batch_size = 32

    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr],
                        verbose=1)

    """
    # save training history
    plt.plot(history.history['my_iou_metric'][1:])
    plt.plot(history.history['val_my_iou_metric'][1:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.savefig('model_loss.png')

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    plt.savefig('train_val_loss.png')
    """

    model = load_model("../output/" + name + ".model")
    preds_valid = predict_result(model,x_valid,img_size_target)

    ## Scoring for last model
    thresholds = np.linspace(0.3, 0.7, 31)
    ious = np.array([iou_metric(y_valid.reshape((-1, img_size_target, img_size_target)), [filter_image(img) for img in preds_valid > threshold]) for threshold in tqdm(thresholds)])

    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()
    plt.savefig('threshold.png')

    del x_train, x_valid, y_train, y_valid, preds_valid
    gc.collect()

    x_test = np.array([(np.array(load_img("../input/test/images/{}.png".format(idx), color_mode = "grayscale"))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)
    preds_test = predict_result(model,x_test,img_size_target)

    t1 = time.time()
    pred_dict = {idx: rle_encode(filter_image(preds_test[i] > threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values))}
    t2 = time.time()

    print("Usedtime = "+ str(t2-t1)+" s")

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    #sub.to_csv('../output/submission.csv')
    return sub


def main():

    # Loading of training/testing ids and depths
    if os.path.isfile('../output/train_df.pkl'):
        train_df = loadpkl('../output/train_df.pkl')
    else:
        train_df, test_df = get_input_data()

    train_df_1 = train_df[train_df.loc[:,'z']<=300]
    test_df_1 = test_df[test_df.loc[:,'z']<=300]
    sub_1 = prediction(train_df_1, test_df_1, 'under_300')

    train_df_2 = train_df[train_df.loc[:,'z']>300]
    test_df_2 = test_df[test_df.loc[:,'z']>300]
    sub_2 = prediction(train_df_2, test_df_2, 'upper_300')

    sub = pd.concat([sub_1, sub_2], axis=0)
    sub.to_csv('../submission.csv')


img_size_target = 101


if __name__ == '__main__':
    main()
