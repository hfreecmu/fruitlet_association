import numpy as np
from inhand_utils import read_dict, write_dict, get_new_basename

input_path = '../preprocess_data/candidate_images.json'
output_path = '../preprocess_data/pairs.json'
num_random = 1000
same_pct = 0.20
#this is only for cross day as same day is still used
min_days_between = 3

input_image_paths = read_dict(input_path)

#first sort all images
fruitlet_dict = {}
for image_path in input_image_paths:
    new_basename = get_new_basename(image_path)
    tag_id_str, date_str = new_basename.split('_')[0:2]
    year_str, month_str, day_str = date_str.split('-')

    tag_year_str = ('_').join([tag_id_str, year_str])
    month_day_str = ('_').join([month_str, day_str])
    
    #create date dict inside tag year dict
    if not tag_year_str in fruitlet_dict:
        fruitlet_dict[tag_year_str] = {}

    #create list inside date dict
    if not month_day_str in fruitlet_dict[tag_year_str]:
        fruitlet_dict[tag_year_str][month_day_str] = []

    #add image  path to list
    fruitlet_dict[tag_year_str][month_day_str].append(image_path)

#now get all pair candidates
#get same day and cross day pairs
cross_pairs = []
same_pairs = []
#this is for duplicates
process_pairs = set()

#I know this looks ugly
for tag_year_str in fruitlet_dict:
    for date_0 in fruitlet_dict[tag_year_str]:
        for image_path_0 in fruitlet_dict[tag_year_str][date_0]:
            for date_1 in fruitlet_dict[tag_year_str]:
                for image_path_1 in fruitlet_dict[tag_year_str][date_1]:
                    pair_0 = '_'.join([image_path_0, image_path_1])
                    pair_1 = '_'.join([image_path_1, image_path_0])

                    if pair_0 in process_pairs:
                        continue
                    if pair_1 in process_pairs:
                        continue

                    process_pairs.add(pair_0)
                    process_pairs.add(pair_1)

                    if date_0 == date_1:
                        if image_path_0 == image_path_1:
                            continue
                        
                        same_pairs.append((image_path_0, image_path_1))
                    else:
                        #check min day
                        day_0 = int(date_0.split('_')[1])
                        day_1 = int(date_1.split('_')[1])

                        if np.abs(day_0 - day_1) < min_days_between:
                            continue
                        
                        cross_pairs.append((image_path_0, image_path_1))

num_same = int(same_pct * num_random)
num_cross = num_random - num_same

same_inds = np.random.choice(len(same_pairs), size=num_same, replace=False)
cross_inds = np.random.choice(len(cross_pairs), size=num_cross, replace=False)

output_pair_paths = []

for ind in same_inds:
    output_pair_paths.append(same_pairs[ind])

for ind in cross_inds:
    output_pair_paths.append(cross_pairs[ind])


write_dict(output_path, output_pair_paths)