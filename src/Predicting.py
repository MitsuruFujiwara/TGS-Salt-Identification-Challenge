
import pandas as pd
import numpy as np
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from keras.models import load_model
from keras.preprocessing.image import load_img

from Utils import predict_result, loadpkl, my_iou_metric, my_iou_metric_2, rle_encode, filter_image, iou_metric, line_notify, upsample, downsample
from Utils import IMG_SIZE_TARGET, IMG_SIZE_ORI, NUM_FOLDS, iou_metric_batch, RLenc
from lovasz_losses_tf import keras_lovasz_softmax
from Learning import bce_dice_loss
from tta_wrapper.tta_wrapper import tta_segmentation

"""
Preprocessingで作成したテストデータ及びLearningで作成したモデルを読み込み、予測結果をファイルとして出力するモジュール。
"""

def main():
    # load datasets
    print('Loading Datasets...')

    test_df = loadpkl('../output/test_df.pkl')
    train_df = loadpkl('../output/train_df.pkl')
    oof_preds = loadpkl('../output/oof_preds.pkl')
    oof_preds = np.array([downsample(img) for img in oof_preds])

    # is_saltが0のデータを除外
#    train_df = train_df[train_df['is_salt']==1]

    x_test = np.array([(upsample(np.array(load_img("../input/test/images/{}.png".format(idx), color_mode = "grayscale")))) / 255 for idx in tqdm(test_df.index)]).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET, 1)
    y_train = np.array(train_df.masks.tolist()).reshape(-1, IMG_SIZE_ORI, IMG_SIZE_ORI, 1)

    # 結果保存用の配列
    sub_preds = np.zeros((x_test.shape[0], IMG_SIZE_ORI, IMG_SIZE_ORI))

    # (128, 128, 3)に変換
    x_test = np.repeat(x_test,3,axis=3)

    print('Generating submission file...')

    # foldごとのモデルを読み込んでsubmission用の予測値を算出
    for n_fold in range(NUM_FOLDS):

        # load model
        model = load_model('../output/UnetResNet34_pretrained_lovasz_'+str(n_fold)+'.model',
                           custom_objects={'my_iou_metric_2': my_iou_metric_2,
#                                           'bce_dice_loss': bce_dice_loss
                                           'keras_lovasz_softmax':keras_lovasz_softmax
                                           })

        # Test time augmnentationを追加しました
        tta_model = tta_segmentation(model, h_flip=True, v_flip= True, rotation_angles=(90, 180, 270),
                                     h_shifts=(-5, 5), merge='mean')

        # testデータの予測値を保存
        sub_preds_single = np.array([downsample(x) for x in tqdm(tta_model.predict(x_test,32).reshape(-1, IMG_SIZE_TARGET, IMG_SIZE_TARGET))])
        sub_preds += sub_preds_single / NUM_FOLDS

        # single modelのsubmission fileを保存（threshold=0）
        pred_dict_single = {idx: RLenc(np.round(sub_preds_single[i] > 0.0)) for i, idx in enumerate(tqdm(test_df.index.values))}
        sub_single = pd.DataFrame.from_dict(pred_dict_single,orient='index')
        sub_single.index.names = ['id']
        sub_single.columns = ['rle_mask']
#        sub_single.loc[~test_df['is_salt'],'rle_mask'] = np.nan # is_saltが0のデータを空欄にします。
        sub_single.to_csv('../output/submission_single_bin_'+str(n_fold)+'.csv')

        print('fold {} finished'.format(n_fold+1))

        del model, sub_preds_single, pred_dict_single, sub_single, tta_model
        gc.collect()

    # thresholdについてはtrain data全てに対するout of foldの結果を使って算出します。
    thresholds_ori = np.linspace(0.3, 0.7, 31)
    thresholds = np.log(thresholds_ori/(1-thresholds_ori)) # lovasz loss用にthresholdの範囲を変更
    ious = np.array([iou_metric_batch(y_train,
                     np.int32(oof_preds > threshold)) for threshold in tqdm(thresholds)])

    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    print('Best IoU score %.6f' % iou_best)

    # threshold確認用の画像を生成
    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()
    plt.savefig('../output/threshold.png')

    t1 = time.time()
    pred_dict = {idx: RLenc(np.round(sub_preds[i] > threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values))}
    t2 = time.time()

    print("Usedtime = "+ str(t2-t1)+" s")

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']

    # is_saltが0のデータを空欄にします。
#    sub.loc[~test_df['is_salt'],'rle_mask'] = np.nan

    sub.to_csv('../output/submission_lovasz.csv')

    # 完了後にLINE通知を送信
    line_notify('Predicting.py finished. Best IoU score is %.6f' % iou_best)

if __name__ == '__main__':
    main()
