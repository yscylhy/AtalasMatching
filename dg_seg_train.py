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
import configparser
from models import DGBoxNet

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, UpSampling2D, concatenate
from keras.models import Model
from keras.losses import binary_crossentropy, mean_squared_error, categorical_crossentropy
from keras.optimizers import Adam
import matplotlib


if __name__ == "__main__":
    sub_folders = ["nissle", "dg_mask", "svg2png"]
    target_size = [280, 200]

    # list(range(22, 31)) + list(range(93, 100)) + list(range(101, 107))
    half_coronal_dg_list = list(range(5, 11))

    # ---- pc path ------
    config = configparser.RawConfigParser()
    config.read('server_config.txt')

    read_path = config.get('path', 'read_path')
    model_path = config.get('path', 'model_path')
    test_img_path = config.get('path', 'test_img_path')
    intensity_aug_num = config.getint('train_paras', 'intensity_aug_num')
    geo_aug_num = config.getint('train_paras', 'geo_aug_num')

    start_time = time.time()

    # ---------------- segmentation model ------------------
    # ---- patch data preparation -----
    print('Start to prepare data at 0 s')
    data_aug = DGAugmentation()
    x_data, y_data = data_aug.load_half_coronal_data(read_path, sub_folders, half_coronal_dg_list,
                                        target_size, intensity_aug_num, geo_aug_num)

    # ---- train the data ----
    print('Start to train model at {:.2f} s'.format(time.time()-start_time))
    dg_segment = DGBoxNet(model_path=model_path)
    dg_segment.train(x_data, y_data, lr=0.001, N_epoch=30, batch_size=16, validation_split=0.05)

    # ---- test the model -----
    print('Start to predict model at {:.2f} s'.format(time.time() - start_time))
    test_img = cv2.imread(test_img_path, 0).astype(np.float) / 255
    test_img = test_img[:, :test_img.shape[1]//2]
    resize_ratio = 280/test_img.shape[0]
    test_img = cv2.resize(test_img, dsize=(target_size[1], target_size[0]))
    prob_map = dg_segment.predict(test_img)
    print(prob_map)
    print('Finished at {:.2f} s'.format(time.time() - start_time))


