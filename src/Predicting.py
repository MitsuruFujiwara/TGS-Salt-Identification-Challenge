
import pandas as pd
import numpy as np
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from keras.models import load_model
from keras.preprocessing.image import load_img

from Utils import predict_result, IMG_SIZE_TARGET, loadpkl, my_iou_metric, rle_encode, filter_image, iou_metric, NUM_FOLDS

"""
Preprocessingで作成したテストデータ及びLearningで作成したモデルを読み込み、予測結果をファイルとして出力するモジュール。
"""

def main():

    test_df = loadpkl('../output/test_df.pkl')
    x_test = np.array([(np.array(load_img("../input/test/images/{}.png".format(idx), color_mode = "grayscale"))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)

    preds_test = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # foldごとのモデルを読み込んでtestデータに対する予測値を保存
    for n_fold in range(NUM_FOLDS):

        # load dataset
        x_valid = loadpkl('../output/x_valid'+str(n_fold)+'.pkl')
        y_valid = loadpkl('../output/y_valid'+str(n_fold)+'.pkl')

        # load model
        model = load_model('../output/unet_best'+str(n_fold)+'.model', custom_objects={'my_iou_metric': my_iou_metric})

        # get prediction for validation data
        preds_valid = predict_result(model, x_valid, IMG_SIZE_TARGET)

        ## Scoring for last model
        thresholds = np.linspace(0.3, 0.7, 31)
        ious = np.array([iou_metric(y_valid.reshape((-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET)),
                        [filter_image(img) for img in preds_valid > threshold]) for threshold in tqdm(thresholds)])

        threshold_best_index = np.argmax(ious)
        iou_best = ious[threshold_best_index]
        threshold_best = thresholds[threshold_best_index]

        # testデータの予測値を保存
        preds_test += predict_result(model, x_test ,IMG_SIZE_TARGET) / NUM_FOLDS

        # 確認用の画像を生成
        plt.plot(thresholds, ious)
        plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
        plt.xlabel("Threshold")
        plt.ylabel("IoU")
        plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
        plt.legend()
        plt.savefig('../output/threshold'+str(n_fold)+'.png')
        plt.close()

        del x_valid, y_valid, preds_valid, model
        gc.collect()

    t1 = time.time()
    pred_dict = {idx: rle_encode(filter_image(preds_test[i] > threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values))}
    t2 = time.time()

    print("Usedtime = "+ str(t2-t1)+" s")

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('../output/submission.csv')

if __name__ == '__main__':
    main()
