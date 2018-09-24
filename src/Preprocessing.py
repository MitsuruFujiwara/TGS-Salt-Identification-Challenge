
import pandas as pd
import numpy as np
import pickle
import warnings
import gc

from keras.models import Model, Sequential
from keras.preprocessing.image import load_img
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add, Flatten, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.initializers import TruncatedNormal, Constant

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from Utils import cov_to_class, save2pkl, IMG_SIZE_ORI, line_notify

"""
提供データを読み込み、データに前処理を施し、モデルに入力が可能な状態でファイル出力するモジュール。
前処理を追加する場合はここに。
"""

warnings.simplefilter(action='ignore', category=FutureWarning)

# binary classification用のモデル
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
    model.add(conv2d(96, 11, strides=(4,4), bias_init=0, input_shape=(IMG_SIZE_ORI, IMG_SIZE_ORI, 1)))
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

def get_binary_labels(train_df, test_df, num_folds):

    # add new label
    train_df.loc[:,'is_salt'] = train_df.loc[:,'masks'].map(lambda x:1 if x.sum()>0 else 0)

    # folds
    folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)

    X = np.array(train_df.images.tolist()).reshape(-1, IMG_SIZE_ORI, IMG_SIZE_ORI, 1)
    Y = train_df['is_salt']

    # test dataをロードしておきます
    x_test = np.array([(np.array(load_img("../input/test/images/{}.png".format(idx), color_mode = "grayscale"))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, IMG_SIZE_ORI, IMG_SIZE_ORI, 1)

    # 予測値の保存用
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['is_salt'])):

        # Create train/validation split stratified by salt coverage
        ids_train, ids_valid = train_df.index.values[train_idx], train_df.index.values[valid_idx]
        x_train, y_train = X[train_idx], Y[train_idx]
        x_valid, y_valid = X[valid_idx], Y[valid_idx]

        # Data augmentation
        # 左右の反転
        x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train = np.append(y_train, y_train, axis=0)

        # 上下の反転
        x_train = np.append(x_train, [np.flipud(x) for x in x_train], axis=0)
        y_train = np.append(y_train, y_train, axis=0)

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

        model = AlexNet()
        model.compile(optimizer=SGD(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_acc', mode = 'max',patience=20, verbose=1)
        model_checkpoint = ModelCheckpoint('../output/AlexNet_binary'+str(n_fold)+'.model',monitor='val_acc',
                                           mode = 'max', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode = 'max',factor=0.2, patience=5, min_lr=0.00001, verbose=1)

        epochs = 5
        batch_size = 32

        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping, model_checkpoint, reduce_lr],
                            verbose=1)

        oof_preds[valid_idx] = model.predict(x_valid).reshape(x_valid.shape[0])
        sub_preds += model.predict(x_test).reshape(x_test.shape[0]) / folds.n_splits

        del img_090, img_180, img_270, tmp_y_train
        del ids_train, ids_valid, x_train, y_train, x_valid, y_valid
        del model, early_stopping, model_checkpoint, reduce_lr, history
        gc.collect()

    train_df['binary_pred'] = oof_preds
    test_df['binary_pred'] = sub_preds

    test_df['is_salt'] = sub_preds>0.5

    return train_df, test_df

def get_input_data():
    # Load training/testing ids and depths
    train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("../input/depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    print("train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Read images and masks
    print("loading images....")
    train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), color_mode = "grayscale")) / 255 for idx in tqdm(train_df.index)]

    print("loading masks....")
    train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), color_mode = "grayscale")) / 255 for idx in tqdm(train_df.index)]

    # Calculating the salt coverage and salt coverage classes
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(IMG_SIZE_ORI, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

    # add new labels
    train_df, test_df = get_binary_labels(train_df, test_df, 3)

    # train_dfとtest_dfをsaveする処理
    save2pkl('../output/train_df.pkl', train_df)
    save2pkl('../output/test_df.pkl', test_df)

    # 完了後にLINE通知を送信
    line_notify('Preprocessing.py finished. train shape: {}, test shape: {}'.format(train_df.shape, test_df.shape))

    return train_df, test_df

if __name__ == '__main__':
    get_input_data()
