
import pandas as pd
import numpy as np
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from keras.models import load_model
from keras.preprocessing.image import load_img

from Utils import predict_result, loadpkl, my_iou_metric, rle_encode, filter_image, iou_metric, line_notify
from Utils import IMG_SIZE_TARGET, NUM_FOLDS

"""
Preprocessingで作成したテストデータ及びLearningで作成したモデルを読み込み、予測結果をファイルとして出力するモジュール。
"""

def main():
    # load datasets
    test_df = loadpkl('../output/test_df.pkl')
    train_df = loadpkl('../output/train_df.pkl')
    oof_preds = loadpkl('../output/oof_preds.pkl')

    x_test = np.array([(np.array(load_img("../input/test/images/{}.png".format(idx), color_mode = "grayscale"))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)
    y_train = np.array(train_df.masks.tolist()).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)

    # 結果保存用の配列
    sub_preds = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # foldごとのモデルを読み込んでsubmission用の予測値を算出
    for n_fold in range(NUM_FOLDS):

        # load model
        model = load_model('../output/unet_best'+str(n_fold)+'.model',
                           custom_objects={'my_iou_metric': my_iou_metric})

        # testデータの予測値を保存
        sub_preds += predict_result(model, x_test ,IMG_SIZE_TARGET) / NUM_FOLDS

        del model
        gc.collect()


    # thresholdについてはtrain data全てに対するout of foldの結果を使って算出します。
    thresholds = np.linspace(0.3, 0.7, 31)
    ious = np.array([iou_metric(y_train.reshape((-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET)),
                    [filter_image(img) for img in oof_preds > threshold]) for threshold in tqdm(thresholds)])

    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    # threshold確認用の画像を生成
    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()
    plt.savefig('../output/threshold.png')

    t1 = time.time()
    pred_dict = {idx: rle_encode(filter_image(preds_test[i] > threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values))}
    t2 = time.time()

    print("Usedtime = "+ str(t2-t1)+" s")

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('../output/submission.csv')

    # 完了後にLINE通知を送信
    line_notify('finished Predicting.py')

if __name__ == '__main__':
    main()
