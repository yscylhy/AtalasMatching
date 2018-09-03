"""
This function do some simple svg processing. Extract the DG mask from the svg file
"""
import os
import os
import shutil
import requests
from svglib import svglib
from reportlab.graphics import renderPM

def get_svg_mask(read_path, write_path, img_name):
    img_idx = img_name.split('.')[0]
    out_svg_name = os.path.join(write_path, img_idx+'.svg')
    out_png_name = os.path.join(write_path, img_idx+'.png')
    read_name = os.path.join(read_path, img_name)

    dg_num = 0
    with open(out_svg_name, 'w') as f_out:
        with open(read_name, 'r') as f_in:
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

    svg_file = svglib.svg2rlg(out_svg_name)
    renderPM.drawToFile(svg_file, out_png_name, fmt="PNG")


def main():
    reference_size_path = "/mnt/hdd/local_data/ABA/half_coronal/nissle"
    read_path = "/mnt/hdd/local_data/ABA/half_coronal/svg"
    write_path = "/mnt/hdd/local_data/ABA/half_coronal/dg_mask"
    img_names = os.listdir(read_path)
    img_names = [img_name for _, img_name in sorted(zip(
        [int(i.split("_")[0]) for i in img_names], img_names))]

    for img_name in img_names:
        print(img_name)
        get_svg_mask(read_path, write_path, img_name)


if __name__ == "__main__":
    main()
