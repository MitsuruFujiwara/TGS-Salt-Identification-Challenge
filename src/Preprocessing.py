
import pandas as pd
import numpy as np

from Utils import cov_to_class

"""
提供データを読み込み、データに前処理を施し、モデルに入力が可能な状態でファイル出力するモジュール。
前処理を追加する場合はここに。
"""

def main():
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
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

    # TODO: train_dfとtest_dfをsaveする処理

    return None

if __name__ == '__main__':
    main()
