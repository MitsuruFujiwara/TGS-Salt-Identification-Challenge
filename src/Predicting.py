
import pandas as pd
import numpy as np
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from keras.models import load_model
from keras.preprocessing.image import load_img

from Utils import predict_result, IMG_SIZE_TARGET, loadpkl, my_iou_metric, rle_encode, filter_image, iou_metric

"""
Preprocessingで作成したテストデータ及びLearningで作成したモデルを読み込み、予測結果をファイルとして出力するモジュール。
"""

def main():
    # load dataset
    x_valid = loadpkl('../output/x_valid.pkl')
    y_valid = loadpkl('../output/y_valid.pkl')
    test_df = loadpkl('../output/test_df.pkl')

    # load model
    model = load_model("../output/unet_best1.model", custom_objects={'my_iou_metric': my_iou_metric})

    # get prediction for validation data
    preds_valid = predict_result(model, x_valid, IMG_SIZE_TARGET)

    ## Scoring for last model
    thresholds = np.linspace(0.3, 0.7, 31)
    ious = np.array([iou_metric(y_valid.reshape((-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET)),
                    [filter_image(img) for img in preds_valid > threshold]) for threshold in tqdm(thresholds)])

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

    del x_valid, y_valid, preds_valid
    gc.collect()

    x_test = np.array([(np.array(load_img("../input/test/images/{}.png".format(idx), color_mode = "grayscale"))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)
    preds_test = predict_result(model, x_test ,IMG_SIZE_TARGET)

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
