
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
from keras.optimizers import SGD, adam
from keras.utils.data_utils import get_file
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from Utils import loadpkl, upsample, downsample, my_iou_metric, my_iou_metric_2, save2pkl, line_notify, predict_result, iou_metric
from Utils import IMG_SIZE_TARGET, NUM_FOLDS, WEIGHTS_PATH, WEIGHTS_PATH_NO_TOP
from Preprocessing import get_input_data
from lovasz_losses_tf import keras_lovasz_softmax

"""
Preprocessingで作成したファイルを読み込み、モデルを学習するモジュール。
学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数も定義してください。
"""

warnings.simplefilter(action='ignore', category=FutureWarning)

##############################################################################
# ResNet50 with pretrained parameter
# https://www.kaggle.com/meaninglesslives/using-resnet50-pretrained-model-in-keras
##############################################################################

def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'keras.., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'keras.., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

#     x = AveragePooling2D((7, 7), name='avg_pool')(x)

#     if include_top:
#         x = Flatten()(x)
#         x = Dense(classes, activation='softmax', name='fc1000')(x)
#     else:
#         if pooling == 'avg':
#             x = GlobalAveragePooling2D()(x)
#         elif pooling == 'max':
#             x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path,by_name=True)
    return model

def get_unet_resnet(resnet_base):

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = UpSampling2D()(conv9)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None, name="prediction")(conv10)
    output_layer = Activation('sigmoid')(output_layer_noActi)

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
        cov_train, cov_test = train_df.coverage.values[train_idx], train_df.coverage.values[valid_idx]
        depth_train, depth_test = train_df.z.values[train_idx], train_df.z.values[valid_idx]

        # Data augmentation
        # 左右の反転
        x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

        # 上下の反転
        x_train = np.append(x_train, [np.flipud(x) for x in x_train], axis=0)
        y_train = np.append(y_train, [np.flipud(x) for x in y_train], axis=0)

        # 画像を回転
        img_090 = [np.rot90(x,1) for x in x_train]
        img_180 = [np.rot90(x,2) for x in x_train]
        img_270 = [np.rot90(x,3) for x in x_train]
        tmp_y_train = y_train

        x_train = np.append(x_train, img_090, axis=0)
        y_train = np.append(y_train, tmp_y_train, axis=0)

        x_train = np.append(x_train, img_180, axis=0)
        y_train = np.append(y_train, tmp_y_train, axis=0)

        x_train = np.append(x_train, img_270, axis=0)
        y_train = np.append(y_train, tmp_y_train, axis=0)

        print("train shape: {}, test shape: {}".format(x_train.shape, y_train.shape))

        # (128, 128, 3)に変換
        x_train = np.repeat(x_train,3,axis=3)
        x_valid = np.repeat(x_valid,3,axis=3)

        # model
        K.clear_session()
        resnet_base = ResNet50(input_shape=(IMG_SIZE_TARGET, IMG_SIZE_TARGET, 3), include_top=False)
        output_layer = get_unet_resnet(resnet_base)
        model_bin = Model(resnet_base.input, output_layer)

        # 最初にlossをbinary_crossentropyにしたモデルを推定します
        if not(os.path.isfile('../output/unet_best_bin_pretrained'+str(n_fold)+'.model')):
            model_bin.compile(loss="binary_crossentropy", optimizer=adam(lr = 0.01), metrics=[my_iou_metric])

            early_stopping_bin = EarlyStopping(monitor='my_iou_metric',
                                               mode = 'max',
                                               patience=10,
                                               verbose=1)

            model_checkpoint_bin = ModelCheckpoint('../output/unet_best_bin_pretrained'+str(n_fold)+'.model',
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
                                        epochs=200,
                                        batch_size=32,
                                        callbacks=[early_stopping_bin, model_checkpoint_bin, reduce_lr],
                                        verbose=1)

            del model_bin, early_stopping_bin, model_checkpoint_bin, history_bin
            gc.collect()

        # binary_crossentropyで推定したモデルをロード
        model = load_model('../output/unet_best_bin_pretrained'+str(n_fold)+'.model',
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

        model_checkpoint = ModelCheckpoint('../output/unet_best_pretrained'+str(n_fold)+'.model',
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
                            epochs=200,
                            batch_size=32,
                            callbacks=[early_stopping, model_checkpoint, reduce_lr],
                            verbose=1)

        # out of foldsの推定結果を保存
        oof_preds[valid_idx] = predict_result(model, x_valid, IMG_SIZE_TARGET)

        # save training history
        plt.plot(history.history['my_iou_metric_2'][1:])
        plt.plot(history.history['val_my_iou_metric_2'][1:])
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

        # foldごとのスコアを送信
        line_notify('fold: %d, train_iou: %.4f val_iou: %.4f'
                    %(n_fold+1, max(history.history['my_iou_metric_2']), max(history.history['val_my_iou_metric_2'])))

        # メモリ節約のための処理
        del ids_train, ids_valid, x_train, y_train, x_valid, y_valid
        del cov_train, cov_test, depth_train, depth_test
        del resnet_base, output_layer, model, early_stopping, model_checkpoint, reduce_lr
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
    train_df = train_df[train_df['is_salt']==1]

    # training
    kfold_training(train_df, NUM_FOLDS, stratified = True, debug= False)

if __name__ == '__main__':
    main()
