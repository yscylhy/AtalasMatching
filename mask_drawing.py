"""
This function do some simple svg processing. Extract the DG mask from the svg file
"""
import os
import os
import shutil
import requests
from svglib import svglib
import cairosvg
from reportlab.graphics import renderPM
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_svg_mask(read_path, write_path, img_idx, output_size):

    tmp_svg_name = os.path.join(write_path, img_idx+'.svg')
    mask_write_name = os.path.join(write_path, 'gt', img_idx+'.png')
    nissle_write_name = os.path.join(write_path, 'nissle', img_idx+'.png')

    png_mask_read_name = os.path.join(read_path, 'svg2png', img_idx+'.png')
    svg_read_name = os.path.join(read_path, 'svg', img_idx+'.svg')
    nissle_read_name = os.path.join(read_path, 'nissle', img_idx+'.jpg')




    dg_num = 0
    with open(tmp_svg_name, 'w') as f_out:
        with open(svg_read_name, 'r') as f_in:
            for _ in range(2):
                cur_line = f_in.readline()
                f_out.write(cur_line)
            cur_line = f_in.readline()
            end_idx = cur_line.find('<path id="')
            f_out.write(cur_line[:end_idx])
            start_record = False

            remain_lines = f_in.readlines()
            for cur_line in remain_lines:
                if ('structure_id="632"' in cur_line) and (not start_record):
                    start_idx = cur_line.find('<path id="')
                    f_out.write(cur_line[start_idx:])
                    start_record = True
                    dg_num += 1
                elif ('structure_id="632"' in cur_line) and start_record:
                    f_out.write(cur_line)
                    dg_num +=1
                elif ('structure_id="632"' not in cur_line) and start_record:
                    if 'structure_id="' in cur_line:
                        end_idx = cur_line.find('<path id="')
                        f_out.write(cur_line[:end_idx])
                        start_record = False
                    else:
                        f_out.write(cur_line)
        f_out.write("</g>")
        f_out.write("</g>")
        f_out.write("</svg>")

    cairosvg.svg2png(url=tmp_svg_name, write_to=mask_write_name)


    whole_mask = cv2.imread(png_mask_read_name, 0)
    dg_mask = cv2.imread(mask_write_name, 0)
    nissle_org = cv2.imread(nissle_read_name, 0)

    # --- resize all the images to the same size
    whole_mask = cv2.resize(whole_mask.astype(np.uint8), dsize=tuple(reversed(nissle_org.shape)),
                            interpolation=cv2.INTER_NEAREST)
    dg_mask = cv2.resize(dg_mask.astype(np.uint8), dsize=tuple(reversed(nissle_org.shape)),
                            interpolation=cv2.INTER_NEAREST)

    # --- resize and crop the nissle and mask for training
    row_index, col_index = np.where(whole_mask!=255)
    start_row = row_index.min()
    end_row = row_index.max()
    start_col = col_index.min()
    end_col = col_index.max()

    dg_mask = dg_mask[start_row: end_row, start_col:end_col]
    nissle_img = nissle_org[start_row: end_row, start_col:end_col]

    resize_ratio = min(output_size[0]/nissle_img.shape[0], output_size[1]/nissle_img.shape[1])
    dg_mask = cv2.resize(dg_mask, dsize=None, fx=resize_ratio, fy=resize_ratio,
                             interpolation=cv2.INTER_NEAREST)
    nissle_img = cv2.resize(nissle_img, dsize=None, fx=resize_ratio, fy=resize_ratio,
                         interpolation=cv2.INTER_NEAREST)

    dg_out = np.zeros(output_size, np.uint8)
    nissle_out = np.ones(output_size, np.uint8)*255
    dg_out[:dg_mask.shape[0], :dg_mask.shape[1]] = dg_mask
    nissle_out[:nissle_img.shape[0], :nissle_img.shape[1]] = nissle_img

    # --- fine largest component
    if dg_out.max() != 0:
        _, contours, _ = cv2.findContours(dg_out.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contour_area = [cv2.contourArea(x) for x in contours]
        max_id = np.argmax(contour_area)
        dg_out = np.zeros(output_size, np.uint8)
        cv2.drawContours(dg_out, contours, max_id, color=(255), thickness=-1)


    cv2.imwrite(mask_write_name, dg_out)
    cv2.imwrite(nissle_write_name, nissle_out)




def main():
    read_path = "/mnt/hdd/local_data/ABA/half_coronal"
    write_path = "/mnt/hdd/local_data/ABA/half_coronal/train_set"
    output_size = [384, 384]

    img_names = os.listdir(os.path.join(read_path, 'svg'))
    imgs_id = [img_name.split(".")[0] for _, img_name in sorted(zip(
        [int(i.split("_")[0]) for i in img_names], img_names))]

    for img_id in imgs_id:
        print(img_id)
        get_svg_mask(read_path, write_path, img_id, output_size)


if __name__ == "__main__":
    main()
