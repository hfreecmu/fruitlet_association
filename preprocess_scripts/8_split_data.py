import os
import random
import shutil

input_dir = '../preprocess_data/pair_annotations'
train_pct = 0.7
val_pct = 0.15
output_dir = '..//datasets'

train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

#TODO fix this
os.makedirs(train_dir)
os.makedirs(val_dir)
os.makedirs(test_dir)

paths = []
for filename in os.listdir(input_dir):
    if not filename.endswith('.json'):
        continue

    paths.append(os.path.join(input_dir, filename))

num_paths = len(paths)
num_train = int(train_pct*num_paths)
num_val = int(val_pct*num_paths)

random.shuffle(paths)

train_paths = paths[0:num_train]
val_paths = paths[num_train:num_train + num_val]
test_paths = paths[num_train + num_val:]


for path in train_paths:
    basename = os.path.basename(path)

    dest = os.path.join(train_dir, basename)
    shutil.copyfile(path, dest)

for path in val_paths:
    basename = os.path.basename(path)

    dest = os.path.join(val_dir, basename)
    shutil.copyfile(path, dest)

for path in test_paths:
    basename = os.path.basename(path)

    dest = os.path.join(test_dir, basename)
    shutil.copyfile(path, dest)

