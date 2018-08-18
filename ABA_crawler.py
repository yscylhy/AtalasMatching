"""
Created by Haoyi Liang on 2018-08-17
This function downloads the raw images and manual labels

The download prototype is form:
    - svg_download_prototype:
        http://help.brain-map.org/display/api/Downloading+and+Displaying+SVG
    - color_coding_prototype, AtlasGroupLabel, OntologyID, AtalsID
        http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies#AtlasDrawingsandOntologies-StructuresAndOntologies
    - nissle_download_prototype
        http://help.brain-map.org/display/api/Downloading+an+Image
"""
import os
import shutil
import requests


# TODO: Download the file that specify the list of all the image names
# TODO: Download the position info about these images


def main():
    write_path = "/mnt/hdd/local_data/ABA"

    AtlasGroupLabel = 28
    OntologyID = 1
    AtalsID = {"coronal": 602630314, "half coronal": 1, "sagittal": 2}
    text_download_prototype = "http://api.brain-map.org/api/v2/data/" +\
        "query.csv?criteria=model::AtlasImage,rma::criteria,atlas_data_set" +\
        "(atlases[id$eq{}]),graphic_objects(graphic_group_label[id$eq{}])," +\
        "rma::options[tabular$eq'sub_images.id'][order$eq'sub_images.id']" +\
        "&num_rows=all&start_row=0"
    svg_download_prototype = "http://api.brain-map.org/api/v2/" +\
        "svg_download/{}?groups={}&quality=100"
    nissle_download_prototype = "http://api.brain-map.org/api/v2/" +\
                                "image_download/{}?downsample=1&atlas={}&quality=100"
    color_coding_prototype = "http://api.brain-map.org/api/v2/structure_graph_download/{}.json"

    # --- get the color coding scheme
    color_coding_url = color_coding_prototype.format(OntologyID)
    color_coding_file_name = os.path.join(write_path, 'color_coding.json')
    with open(color_coding_file_name, 'wb') as f:
        f.write(requests.get(color_coding_url).content)

    for atlas_name in AtalsID:
        print(atlas_name)
        # --- get the list of images with ground truth
        url = text_download_prototype.format(AtalsID[atlas_name], AtlasGroupLabel)

        with open(os.path.join(write_path, atlas_name+'.txt'), 'wb') as f:
            img_id_with_label = requests.get(url).content
            f.write(img_id_with_label)

        if os.path.exists(os.path.join(write_path, atlas_name)):
            shutil.rmtree(os.path.join(write_path, atlas_name))
        os.mkdir(os.path.join(write_path, atlas_name))

        # --- download all the images with ground truth
        img_lists = img_id_with_label.split('\n')
        for ord_id, img_id in enumerate(img_lists[1:]):
            print(ord_id)
            svg_url = svg_download_prototype.format(img_id, AtlasGroupLabel)
            svg_file_name = os.path.join(write_path, atlas_name,
                                         '{}_{}.svg'.format(ord_id, img_id))
            jpg_url = nissle_download_prototype.format(img_id, AtalsID[atlas_name])
            jpg_file_name = os.path.join(write_path, atlas_name,
                                         '{}_{}.jpg'.format(ord_id, img_id))

            with open(svg_file_name, 'wb') as f:
                f.write(requests.get(svg_url).content)
            with open(jpg_file_name, 'wb') as f:
                f.write(requests.get(jpg_url).content)


if __name__ == "__main__":
    main()
