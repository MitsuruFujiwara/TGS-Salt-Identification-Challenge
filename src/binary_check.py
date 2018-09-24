
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import time
from tqdm import tqdm

from keras.models import Model, Sequential
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add, Flatten, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD
from keras import backend as K
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.initializers import TruncatedNormal, Constant

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

    """
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
    """
    uconv1 = Dropout(DropoutRatio/2)(convm)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)

    output_layer = Flatten()(output_layer)
    output_layer = Dense(1, activation='softmax')(output_layer)

    return output_layer

# レシートのとき使ったAlex Netを改変して使います
def conv2d(filters, kernel_size, strides=1, bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        activation='relu',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

def dense(units, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
        units,
        activation='tanh',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

def AlexNet():
    model = Sequential()

    # 第1畳み込み層
    model.add(conv2d(96, 11, strides=(4,4), bias_init=0, input_shape=(img_size_target, img_size_target, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第２畳み込み層
    model.add(conv2d(256, 5, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第３~5畳み込み層
    model.add(conv2d(384, 3, bias_init=0))
    model.add(conv2d(384, 3, bias_init=1))
    model.add(conv2d(256, 3, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 密結合層
    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))
    model.add(dense(4096))
    model.add(Dropout(0.5))

    # 読み出し層
    model.add(Dense(1, activation='sigmoid'))

    return model

def prediction(train_df, test_df, name):
    # Create train/validation split stratified by salt coverage
    train_df.loc[:,'is_salt'] = train_df.loc[:,'masks'].map(lambda x:1 if x.sum()>0 else 0)

    ids_train, ids_valid, x_train, x_valid, y_train, y_valid,\
    cov_train, cov_test, depth_train, depth_test = train_test_split(train_df.index.values,
                                                                    np.array(train_df.images.tolist()).reshape(-1, img_size_target, img_size_target, 1),
                                                                    #np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1),
                                                                    train_df.is_salt,
                                                                    train_df.coverage.values,
                                                                    train_df.z.values,
                                                                    test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)

    #Data augmentation
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, y_train, axis=0)
    print("train shape: {}, test shape: {}".format(x_train.shape, y_train.shape))

    # model
#    input_layer = Input((img_size_target, img_size_target, 1))
#    output_layer = build_model(input_layer, 16,0.5)

#    model = Model(input_layer, output_layer)
#    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    # Alex Netを使います
    model = AlexNet()
    model.compile(optimizer=SGD(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_acc', mode = 'max',patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint("../output/" + name + ".model",monitor='val_acc',
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode = 'max',factor=0.2, patience=5, min_lr=0.00001, verbose=1)
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

#    preds_valid = predict_result(model,x_valid,img_size_target)

    # test dataのカラムにモデル予測値を保存
    x_test = np.array([(np.array(load_img("../input/test/images/{}.png".format(idx), color_mode = "grayscale"))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)
    test_df.loc[:,'is_salt']= model.predict(x_test)

    del x_train, x_valid, y_train, y_valid, preds_valid
    gc.collect()

    # is_saltカラムを追加したdfを保存します
    save2pkl('../output/train_df.pkl', train_df)
    save2pkl('../output/test_df.pkl', test_df)

    return


def main():

    # Loading of training/testing ids and depths
    if os.path.isfile('../output/train_df.pkl') and os.path.isfile('../output/test_df.pkl'):
        train_df = loadpkl('../output/train_df.pkl')
        test_df = loadpkl('../output/test_df.pkl')
    else:
        train_df, test_df = get_input_data()

    prediction(train_df, test_df, 'AlexNet_binary')

    """
    train_df = train_df[train_df.loc[:,'z']<=300]
    test_df = test_df[test_df.loc[:,'z']<=300]
    sub = prediction(train_df_1, test_df_1, 'under_300')

    sub.to_csv('../submission.csv')
    """


img_size_target = 101


if __name__ == '__main__':
    main()
