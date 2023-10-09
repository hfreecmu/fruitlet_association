import os
import cv2
from inhand_utils import write_dict

#https://pupil-apriltags.readthedocs.io/en/stable/api.html
#https://github.com/pupil-labs/apriltags
from pupil_apriltags import Detector as AprilTagDetector

###these are taken from get_training_images.py

inhand_dir_2021 = '/media/frc-ag-3/umass_1/umass_2021_data/umass_2021_bags'
inhand_dir_2023 = '/media/frc-ag-3/umass_1/umass_2023_data/in-hand_images/rectified_images/70_clusters'

filter_dict = {inhand_dir_2021: (True, {245, 250, 170, 172, 173, 174, 176,
                                        177, 190, 193, 229, 238, 247, 251,
                                        8, 15, 42, 48, 51, 54, 58,
                                        60, 62, 67, 69, 71, 81, 82,
                                        95, 97, 100, 105, 106, 120, 121,
                                        125, 128, 137, 154, 160, 161, 163}),
               inhand_dir_2023: (False, set())
               }

training_dirs = [inhand_dir_2021,
                 inhand_dir_2023]
###

output_path = '../preprocess_data/candidate_images.json'


#get all valid tag dirs
tag_dirs = set()
for training_dir in training_dirs:
    should_filter, target_tags = filter_dict[training_dir]

    for tag_dirname in os.listdir(training_dir):
        tag_dir = os.path.join(training_dir, tag_dirname)

        if not os.path.isdir(tag_dir):
            continue

        tag_id = int(tag_dirname.split('_')[0])

        if should_filter:
            if not tag_id in target_tags:
                continue

        tag_dirs.add(tag_dir)

image_list = []
for tag_dir in tag_dirs:
    image_dir = os.path.join(tag_dir, 'COLOR')

    if not os.path.exists(image_dir):
        raise RuntimeError('No COLOR dir in ' + tag_dir)

    for filename in os.listdir(image_dir):
        if not filename.endswith('.png'):
            continue

        if not 'LEFT' in filename:
            continue

        image_path = os.path.join(image_dir, filename)
        image_list.append(image_path)

#now go through images and detect april tags
#TODO can migrate to inhand_utils detect april tags when ready
detector = AprilTagDetector()
filtered_image_list = []
num_unfiltered = len(image_list)
count = 0
for image_path in image_list:
    count += 1

    if count % 100 == 0:
        print('Done:', count, 'out of', num_unfiltered, 
              'with', len(filtered_image_list), 'found')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    color_dir = os.path.dirname(image_path)
    tag_dir = os.path.dirname(color_dir)
    tag_dirname = os.path.basename(tag_dir)
    
    tag_id = int(tag_dirname.split('_')[0])

    results = detector.detect(gray_image)
    
    num_found = 0
    for result in results:
        if result.tag_id == tag_id:
            num_found += 1

            if num_found > 1:
                break

    #only add tag if found exactly once
    if num_found == 1:
        filtered_image_list.append(image_path)

write_dict(output_path, filtered_image_list)