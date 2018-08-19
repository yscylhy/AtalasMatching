"""
Created by Haoyi Liang on 2018-08-18
This function train the dentate gyrus segmentor
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
import time
from keras.utils import to_categorical
import cv2
import tensorflow as tf
import keras.backend as K
from data_augmentation import DGAugmentation
from models import SegFCNet

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D, concatenate
from keras.models import Model
from keras.losses import binary_crossentropy, mean_squared_error, categorical_crossentropy
from keras.optimizers import Adam
import matplotlib



if __name__ == "__main__":
    sub_folders = ["nissle", "svg2png"]
    target_size = [304, 400]
    dg_color = [61, 168, 102]  # BGR. HEX: 0x66A83D
    intensity_aug_num = 8
    geo_aug_num = 16
    # list(range(22, 31)) + list(range(93, 100)) + list(range(101, 107))
    coronal_dg_list = list(range(22, 31)) + list(range(93, 100)) + list(range(101, 107))

    # ---- pc path ------
    read_path = "/mnt/hdd/local_data/ABA/coronal"
    model_path = "/mnt/hdd/local_data/ABA_DG_model"
    test_img_path = "/mnt/hdd/local_data/ABA/coronal/nissle/21_576986851.jpg"

    # ---- server path ------
    # read_path = "/mnt/hdd2/ABA/coronal"
    # model_path = "/mnt/hdd2/ABA_DG_model"
    # test_img_path = "/mnt/hdd2/ABA/coronal/nissle/21_576986851.jpg"

    start_time = time.time()

    # ---------------- segmentation model ------------------
    # ---- patch data preparation -----
    # print('Start to prepare data at 0 s')
    # data_aug = DGAugmentation()
    # # x_data, y_data = data_aug.get_center_regression_label(img_path, gt_path, intensity_aug_num=64, geo_aug_num=64)
    # x_data, y_data = data_aug.load_data(read_path, sub_folders, coronal_dg_list,
    #                                     target_size, dg_color, intensity_aug_num, geo_aug_num)
    # ---- train the data ----
    # print('Start to train model at {:.2f} s'.format(time.time()-start_time))
    dg_segment = SegFCNet(model_path=model_path)
    # dg_segment.train(x_data, y_data, lr=0.0001, N_epoch=10, batch_size=16, validation_split=0.01)

    # ---- test the model -----
    print('Start to predict model at {:.2f} s'.format(time.time() - start_time))
    test_img = cv2.imread(test_img_path, 0).astype(np.float) / 255
    resize_ratio = 300/test_img.shape[0]
    test_img = cv2.resize(test_img, dsize=None, fx=resize_ratio, fy=resize_ratio)
    trim_row = test_img.shape[0] % 8
    trim_col = test_img.shape[1] % 8
    test_img = test_img[trim_row:, trim_col:]
    prob_map = dg_segment.predict(test_img)
    print('Finished at {:.2f} s'.format(time.time() - start_time))


