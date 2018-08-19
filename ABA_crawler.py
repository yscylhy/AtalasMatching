"""
Created by Haoyi Liang on 2018-08-17
This function downloads the raw images and manual labels

The download prototype is from:
    - svgGT_download_prototype:
        http://help.brain-map.org/display/api/Downloading+and+Displaying+SVG
    - color_coding_prototype, jpgGT_download_protype, AtlasGroupLabel, OntologyID, AtalsID
        http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies#AtlasDrawingsandOntologies-StructuresAndOntologies
    - nissle_download_prototype
        http://help.brain-map.org/display/api/Downloading+an+Image
"""

import os
import shutil
import requests
import time
from svglib import svglib
from reportlab.graphics import renderPM

# TODO: Download the file that specify the list of all the image names
# TODO: Download the position info about these images


def load_ABA_info():
    # --- these info is found on the ABA website
    AtlasGroupLabel = 28
    OntologyID = 1
    AtalsID = {"coronal": 602630314, "half_coronal": 1, "sagittal": 2}
    text_download_prototype = "http://api.brain-map.org/api/v2/data/" +\
        "query.csv?criteria=model::AtlasImage,rma::criteria,atlas_data_set" +\
        "(atlases[id$eq{}]),graphic_objects(graphic_group_label[id$eq{}])," +\
        "rma::options[tabular$eq'sub_images.id'][order$eq'sub_images.id']" +\
        "&num_rows=all&start_row=0"
    svgGT_download_prototype = "http://api.brain-map.org/api/v2/" +\
        "svg_download/{}?groups={}&quality=100"
    jpgGT_download_protype = "http://api.brain-map.org/api/v2/atlas_image_download"+\
                             "/{}?downsample=1&annotation=true&atlas={}&quality=100"
    nissle_download_prototype = "http://api.brain-map.org/api/v2/" +\
                                "image_download/{}?downsample=1&atlas={}&quality=100"
    color_coding_prototype = "http://api.brain-map.org/api/v2/structure_graph_download/{}.json"
    return AtlasGroupLabel, OntologyID, AtalsID, text_download_prototype, \
           svgGT_download_prototype, jpgGT_download_protype, \
           nissle_download_prototype, color_coding_prototype


def svg2png(input_name, output_name):
    svg_file = svglib.svg2rlg(input_name)
    renderPM.drawToFile(svg_file, output_name, fmt="PNG")
    return None


def main():
    write_path = "/mnt/hdd/local_data/ABA"
    sub_folers = ['svg', 'nissle', 'jpgGT', 'svg2png']

    # --- load the info from official ABA website
    AtlasGroupLabel, OntologyID, AtalsID, text_download_prototype, svgGT_download_prototype, jpgGT_download_protype, \
    nissle_download_prototype, color_coding_prototype = load_ABA_info()

    # --- get the color coding scheme
    color_coding_url = color_coding_prototype.format(OntologyID)
    color_coding_file_name = os.path.join(write_path, 'color_coding.json')
    with open(color_coding_file_name, 'wb') as f:
        f.write(requests.get(color_coding_url).content)

    # --- download from the ABA official website
    for atlas_name in AtalsID:
        start_time = time.time()
        print("{} Start Download".format(atlas_name))
        # --- get the list of images with ground truth
        url = text_download_prototype.format(AtalsID[atlas_name], AtlasGroupLabel)

        with open(os.path.join(write_path, atlas_name+'.txt'), 'wb') as f:
            img_id_with_label = requests.get(url).content
            f.write(img_id_with_label)

        # --- make all the sub-folders
        if os.path.exists(os.path.join(write_path, atlas_name)):
            shutil.rmtree(os.path.join(write_path, atlas_name))
        os.mkdir(os.path.join(write_path, atlas_name))
        for sub_folder in sub_folers:
            os.mkdir(os.path.join(write_path, atlas_name, sub_folder))

        # --- download all the images with ground truth
        img_lists = img_id_with_label.decode('utf-8').split('\n')
        for ord_id, img_id in enumerate(img_lists[1:-1]):
            cur_time = time.time()
            print('{} of {}({:.1f}s)'.format(ord_id, len(img_lists)-1,
                  cur_time-start_time))
            svg_url = svgGT_download_prototype.format(img_id, AtlasGroupLabel)
            svg_file_name = os.path.join(write_path, atlas_name, 'svg',
                                         '{}_{}.svg'.format(ord_id, img_id))
            svg2png_name = os.path.join(write_path, atlas_name, 'svg2png',
                                         '{}_{}.png'.format(ord_id, img_id))
            nissle_url = nissle_download_prototype.format(img_id, AtalsID[atlas_name])
            nissle_file_name = os.path.join(write_path, atlas_name, 'nissle',
                                         '{}_{}.jpg'.format(ord_id, img_id))
            jpgGT_url = jpgGT_download_protype.format(img_id, AtalsID[atlas_name])
            jpgGT_name = os.path.join(write_path, atlas_name, 'jpgGT',
                                         '{}_{}.jpg'.format(ord_id, img_id))

            with open(svg_file_name, 'wb') as f:
                f.write(requests.get(svg_url).content)
            svg2png(svg_file_name, svg2png_name)
            with open(nissle_file_name, 'wb') as f:
                f.write(requests.get(nissle_url).content)
            with open(jpgGT_name, 'wb') as f:
                f.write(requests.get(jpgGT_url).content)

        cur_time = time.time()
        print("{} done with {:.1f}s".format(atlas_name, cur_time-start_time))


if __name__ == "__main__":
    main()
