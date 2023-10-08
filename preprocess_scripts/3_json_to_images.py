import os
import json
from inhand_utils import read_dict, get_new_basename

def read_dict(path):
    with open(path) as f:
        data = json.load(f)
    return data

json_path = '../preprocess_data/pairs.json'
output_dir = '../preprocess_data/pair_images'
use_softlink = False

images_list = read_dict(json_path)

image_dict = set()
for image_path_0, image_path_1 in images_list:
    for image_path in [image_path_0, image_path_1]:
        new_basename = get_new_basename(image_path)

        if new_basename in image_dict:
            continue

        image_dict.add(new_basename)
        
        right_path = image_path.replace('LEFT', 'RIGHT')
        right_basename = new_basename.replace('LEFT', 'RIGHT')

        left_output_path = os.path.join(output_dir, new_basename)
        right_output_path = os.path.join(output_dir, right_basename)

        input_paths = [image_path, right_path]
        output_paths = [left_output_path, right_output_path]

        for i in range(2):
            src = input_paths[i]
            dest = output_paths[i]

            if use_softlink:
                command = 'ln -s ' + src + ' ' + dest
            else:
                command = 'cp ' + src + ' ' + dest

            os.system(command)


