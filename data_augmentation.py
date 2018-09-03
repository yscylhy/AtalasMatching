"""
created by Haoyi Liang on 2018-08-18
This function reads the original image and the ground truth label
"""

import scipy.io as spio
import time
import numpy as np
import scipy.ndimage as spyimg
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.image as mpimg
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.patches as patches

class DGAugmentation:

    def __init__(self):
        self.target_size = None

    def check_valid_imgs(self, read_path, target_color):
        img_list = os.listdir(read_path)
        # --- sort the image name
        img_list = [img_name for _, img_name in sorted(zip(
            [int(i.split("_")[0]) for i in img_list], img_list))]

        for img_name in img_list:
            label_img = cv2.imread(os.path.join(read_path,img_name), 1)
            mask = np.logical_and(label_img[:, :, 0] == target_color[0],
                                  label_img[:, :, 1] == target_color[1],
                                  label_img[:, :, 2] == target_color[2])
            # --- rendered png from the svg may change the color on some boundaries
            kernel = np.ones((3, 3), np.uint8)
            opened_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

            if opened_mask.sum() != 0:
                print("{} contain target color".format(img_name))

    def load_half_coronal_data(self, read_path, sub_folders, coronal_dg_list, target_size,
                               intensity_aug_num, geo_aug_num):
        self.target_size = target_size
        img_list = os.listdir(os.path.join(read_path, sub_folders[0]))
        # --- sort the image name
        img_list = [img_name for _, img_name in sorted(zip(
            [int(i.split("_")[0]) for i in img_list], img_list))]
        read_list = [img_list[i] for i in coronal_dg_list]

        x_data = np.zeros([len(read_list)] + target_size)
        y_data = np.zeros([len(read_list)] + target_size)
        for idx, img_name in enumerate(read_list):
            [img_idx, img_format] = img_name.split('.')
            x_data[idx, :, :], y_data[idx, :, :] = self.get_one_pair(read_path, sub_folders, img_idx)

        # y_data = np.expand_dims(y_data, axis=3)
        # y_data = self.get_bounding_box(y_data)
        # for idx, img in enumerate(x_data):
        #     fig, ax = plt.subplots(1)
        #     ax.imshow(img)
        #     start_row = y_data[idx, 0]
        #     start_col = y_data[idx, 1]
        #     row_len = y_data[idx, 2]
        #     col_len = y_data[idx, 3]
        #     rect = patches.Rectangle((start_col, start_row), col_len, row_len,
        #                              linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        #     plt.close()

        aug_imgs, aug_labels = self.intensity_augmentation(x_data, y_data, aug_num=intensity_aug_num,
                                                           gamma_bound=(0.9, 1.1), intensity_bound=(0.99, 1))
        aug_merged_imgs = self.geometric_augmentation(aug_imgs, aug_labels,
                                                      round_labels=True, aug_number=geo_aug_num)

        x_data = np.expand_dims(aug_merged_imgs[:, :, :, 0], axis=3)
        y_data = np.expand_dims(aug_merged_imgs[:, :, :, 2], axis=3)

        y_data = self.get_bounding_box(y_data)

        return x_data, y_data

    def load_coronal_data(self, read_path, sub_folders, coronal_dg_list, target_size, dg_color, intensity_aug_num, geo_aug_num):
        self.target_size = target_size
        self.dg_color = dg_color

        img_list = os.listdir(os.path.join(read_path, sub_folders[0]))
        # --- sort the image name
        img_list = [img_name for _, img_name in sorted(zip(
            [int(i.split("_")[0]) for i in img_list], img_list))]
        read_list = [img_list[i] for i in coronal_dg_list]

        x_data = np.zeros([len(read_list)] + target_size)
        y_data = np.zeros([len(read_list)] + target_size)
        for idx, img_name in enumerate(read_list):
            [img_idx, img_format] = img_name.split('.')
            x_data[idx, :, :], y_data[idx, :, :] = self.get_one_pair(read_path, sub_folders, img_idx)

        aug_imgs, aug_labels = self.intensity_augmentation(x_data, y_data, aug_num=intensity_aug_num,
                                                           gamma_bound=(0.5, 1), intensity_bound=(2/3, 1))
        aug_merged_imgs = self.geometric_augmentation(aug_imgs, aug_labels,
                                                      round_labels=True, aug_number=geo_aug_num)

        x_data = np.expand_dims(aug_merged_imgs[:, :, :, 0], axis=3)
        y_data = np.expand_dims(aug_merged_imgs[:, :, :, 2], axis=3)
        return x_data, y_data

    def get_one_pair(self, dir_path, sub_folders, img_idx):
        target_size = self.target_size
        nissle_img = cv2.imread(os.path.join(dir_path, sub_folders[0], img_idx + '.jpg'), 0)
        label_img = cv2.imread(os.path.join(dir_path, sub_folders[1], img_idx + '.png'), 0)
        whole_label_img = cv2.imread(os.path.join(dir_path, sub_folders[2], img_idx + '.png'), 0)

        # --- first resize label_img to the nissle_img
        if nissle_img.shape[0] / label_img.shape[0] != nissle_img.shape[1] / label_img.shape[1]:
            print("ERROR: mask and nissle images do not match!")
            return None

        # --- extract the dg_mask
        dg_mask = (label_img != 255)
        dg_mask = cv2.resize(dg_mask.astype(np.uint8), dsize=tuple(reversed(nissle_img.shape)),
                             interpolation=cv2.INTER_NEAREST)
        whole_mask = whole_label_img != 255
        whole_mask = cv2.resize(whole_mask.astype(np.uint8), dsize=tuple(reversed(nissle_img.shape)),
                                interpolation=cv2.INTER_NEAREST)

        # --- get the largest connected component
        _, contours, _ = cv2.findContours(dg_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contour_area = [cv2.contourArea(x) for x in contours]
        max_id = np.argmax(contour_area)
        dg_mask = np.zeros(dg_mask.shape, np.uint8)
        cv2.drawContours(dg_mask, contours, max_id, color=(255), thickness=-1)

        # --- crop the mask and nissle images
        row_index, col_index = np.where(whole_mask)
        start_row = row_index.min()
        end_row = row_index.max()
        start_col = col_index.min()
        end_col = col_index.max()

        dg_mask = dg_mask[start_row: end_row, start_col:end_col]
        nissle_img = nissle_img[start_row: end_row, start_col:end_col]

        # --- resize both mask and nissle images to the target size
        resize_width = target_size[0] / nissle_img.shape[0]
        resize_heigth = target_size[1] / nissle_img.shape[1]
        resize_ratio = min(resize_heigth, resize_width)
        resized_nissle_img = cv2.resize(nissle_img.astype(np.float), dsize=None, fx=resize_ratio, fy=resize_ratio,
                                        interpolation=cv2.INTER_NEAREST)
        resized_dg_mask = cv2.resize(dg_mask.astype(np.float), dsize=None, fx=resize_ratio, fy=resize_ratio,
                                     interpolation=cv2.INTER_NEAREST)

        x_data = np.ones(target_size)*255
        y_data = np.zeros(target_size)
        x_data[:resized_nissle_img.shape[0], :resized_nissle_img.shape[1]] = resized_nissle_img
        y_data[:resized_dg_mask.shape[0], :resized_dg_mask.shape[1]] = resized_dg_mask
        return x_data, y_data

    @staticmethod
    def get_bounding_box(imgs):
        y_label = np.zeros([imgs.shape[0], 4])
        for idx, img in enumerate(imgs):
            row_index, col_index = np.where(img[:, :, 0])
            start_row = row_index.min()
            end_row = row_index.max()
            start_col = col_index.min()
            end_col = col_index.max()

            y_label[idx] = [start_row, start_col, end_row-start_row, end_col - start_col]
        return y_label


    @staticmethod
    def intensity_augmentation(org_img, org_label, aug_num=32, gamma_bound=(0.01, 1), intensity_bound=(1/3, 1)):
        # ---- gamma correction and intensity suppression

        aug_imgs = np.repeat(org_img, repeats=aug_num, axis=0)
        aug_labels = np.repeat(org_label, repeats=aug_num, axis=0)

        for i in range(aug_imgs.shape[0]):
            gamma_value = np.random.uniform(gamma_bound[0], gamma_bound[1])
            intensity_suppression = np.random.uniform(intensity_bound[0], intensity_bound[1])

            tmp_aug_img = aug_imgs[i] ** gamma_value
            tmp_aug_img = (tmp_aug_img - np.min(tmp_aug_img)) / (np.max(tmp_aug_img) - np.min(tmp_aug_img))
            tmp_aug_img = tmp_aug_img * intensity_suppression

            aug_imgs[i] = tmp_aug_img

        return aug_imgs, aug_labels

    @staticmethod
    def geometric_augmentation(org_img, org_label, round_labels=True, aug_number=5, rotation_range=360, h_flip=True,
                               v_flip=True, fill_mode='constant'):

        merge_img = np.zeros(org_img.shape + (3,))
        merge_img[:, :, :, 0] = org_img
        merge_img[:, :, :, 2] = org_label
        input_img_num = org_img.shape[0]

        aug_merge_imgs = np.zeros((input_img_num*aug_number, ) + merge_img.shape[1:])

        data_gen = ImageDataGenerator(
            rotation_range=rotation_range,
            horizontal_flip=h_flip,
            vertical_flip=v_flip,
            fill_mode=fill_mode,
            cval=0)

        for _idx, batch in enumerate(data_gen.flow(merge_img, batch_size=input_img_num)):
            aug_merge_imgs[_idx*input_img_num: (_idx+1)*input_img_num] = batch
            if _idx == aug_number-1:
                break

        # categorical label should be rounded here
        if round_labels:
            aug_merge_imgs[:, :, :, 2] = np.round(aug_merge_imgs[:, :, :, 2])

        return aug_merge_imgs


def main():
    # --- coronal images that contain dg: list(range(22, 31)) + list(range(93, 100)) + list(range(101, 107))
    # --- half_coronal images that contain dg: list(range(1, 16)) + list(range(48, 60)) + list(range(123, 124))
    read_path = "/mnt/hdd/local_data/ABA/half_coronal"
    sub_folders = ["nissle", "svg2png"]
    target_size = [300, 225]
    dg_color = [61, 168, 102]  # BGR. HEX: 0x66A83D
    intensity_aug_num = 2
    geo_aug_num = 2

    # coronal_dg_list = list(range(22, 31)) + list(range(93, 100)) + list(range(101, 107))
    half_coronal_dg_list = [9, 10]

    data_aug = DGAugmentation()
    # data_aug.check_valid_imgs(read_path, dg_color)

if __name__ == "__main__":
    main()

