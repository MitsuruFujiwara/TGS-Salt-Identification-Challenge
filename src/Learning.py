
import pandas as pd
import numpy as np
import Preprocessing

from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
from Utils import loadpkl

"""
Preprocessingで作成したファイルを読み込み、モデルを学習するモジュール。
学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数も定義してください。
"""

IMG_SIZE_TARGET = 128

def main():
    # load saved dataset
    train_df = loadpkl('../output/train_df.pkl')
    test_df = loadpkl('../output/test_df.pkl')

    # とりあえず直接関数を呼ぶ形式に
#    train_df, test_df = Preprocessing.main()

    # Create train/validation split stratified by salt coverage
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid,\
    cov_train, cov_test, depth_train, depth_test = train_test_split(train_df.index.values,
                                                                    np.array(train_df.images.tolist()).reshape(-1, img_size_target, img_size_target, 1),
                                                                    np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1),
                                                                    train_df.coverage.values,
                                                                    train_df.z.values,
                                                                    test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)

    # set model
    model = ResNet50(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=(IMG_SIZE_TARGET,IMG_SIZE_TARGET,3),
                     pooling=None,
                     classes=1000)

    get_unet_resnet(input_shape=(IMG_SIZE_TARGET,IMG_SIZE_TARGET,3))

if __name__ == '__main__':
    main()
