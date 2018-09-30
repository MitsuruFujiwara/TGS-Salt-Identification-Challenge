
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
from keras import optimizers
from Utils import loadpkl, upsample, downsample, my_iou_metric, save2pkl, line_notify, predict_result, iou_metric, my_iou_metric_2
from lovasz_losses_tf import lovasz_grad, lovasz_hinge, lovasz_hinge_flat, flatten_binary_scores, lovasz_loss
from Preprocessing import get_input_data

"""
Preprocessingで作成したファイルを読み込み、モデルを学習するモジュール。
学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数も定義してください。
"""

warnings.simplefilter(action='ignore', category=FutureWarning)

ACTIVATION = "relu"

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

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

    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)
    
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

    model1 = Model(input_layer, output_layer)

    c = optimizers.adam(lr = 0.01)
    model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])

    #early_stopping = EarlyStopping(monitor='my_iou_metric', mode = 'max',patience=10, verbose=1)
    save_model_name = name + '.model'
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='my_iou_metric', 
                                    mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.5, patience=6, min_lr=0.0001, verbose=1)

    epochs = 55
    batch_size = 32
    history = model1.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[ model_checkpoint,reduce_lr], 
                        verbose=2)
    
    model1 = load_model(save_model_name,custom_objects={'my_iou_metric': my_iou_metric})
    # remove layter activation layer and use losvasz loss
    input_x = model1.layers[0].input

    output_layer = model1.layers[-1].input
    model = Model(input_x, output_layer)
    c = optimizers.adam(lr = 0.01)

    # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
    # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
    model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])


    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2', 
                                    mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    epochs = 50
    batch_size = 32

    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[ model_checkpoint,reduce_lr,early_stopping], 
                        verbose=2)
    
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


    model = load_model(save_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                    'lovasz_loss': lovasz_loss})
    def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
        x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
        preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
        preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
        preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
        return preds_test/2

    preds_valid = predict_result(model,x_valid,img_size_target)

    # src: https://www.kaggle.com/aglotero/another-iou-metric
    def iou_metric(y_true_in, y_pred_in, print_table=False):
        labels = y_true_in
        y_pred = y_pred_in


        true_objects = 2
        pred_objects = 2

        #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
        temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
    #     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
        #print(temp1)
        intersection = temp1[0]
        #print("temp2 = ",temp1[1])
        #print(intersection.shape)
    # print(intersection)
        # Compute areas (needed for finding the union between all objects)
        #print(np.histogram(labels, bins = true_objects))
        area_true = np.histogram(labels,bins=[0,0.5,1])[0]
        #print("area_true = ",area_true)
        area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection
    
        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        intersection[intersection == 0] = 1e-9
        
        union = union[1:,1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Precision helper function
        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1   # Correct objects
            false_positives = np.sum(matches, axis=0) == 0  # Missed objects
            false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
            tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            return tp, fp, fn

        # Loop over IoU thresholds
        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)
        
        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return np.mean(prec)

    def iou_metric_batch(y_true_in, y_pred_in):
        batch_size = y_true_in.shape[0]
        metric = []
        for batch in range(batch_size):
            value = iou_metric(y_true_in[batch], y_pred_in[batch])
            metric.append(value)
        return np.mean(metric)

    ## Scoring for last model, choose threshold by validation data 
    thresholds_ori = np.linspace(0.3, 0.7, 31)
    # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
    thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

    # ious = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
    # print(ious)
    ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm(thresholds)])
    print(ious)

    # instead of using default 0 as threshold, use validation data to find the best threshold.
    threshold_best_index = np.argmax(ious) 
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()

    """
    used for converting the decoded image to rle mask
    Fast compared to previous one
    """
    def rle_encode(im):
        '''
        im: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = im.flatten(order = 'F')
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    x_test = np.array([(np.array(load_img("../input/test/images/{}.png".format(idx), grayscale = True))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)
    preds_test = predict_result(model,x_test,img_size_target)

    t1 = time.time()
    pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values))}
    t2 = time.time()

    print(f"Usedtime = {t2-t1} s")

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']

    return sub


def main():

    # Loading of training/testing ids and depths
    if os.path.isfile('../output/train_df.pkl'):
        train_df = loadpkl('../output/train_df.pkl')
        test_df = loadpkl('../output/test_df.pkl')
    else:
        train_df, test_df = get_input_data(add_new_label=False)

    sub = prediction(train_df, test_df, 'test')

    sub.to_csv('../submission.csv')


img_size_target = 101


if __name__ == '__main__':
    main()
