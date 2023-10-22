import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
from utils.utils import read_dict
import PIL.Image as Image


# ROW_MEAN = 539.5
# ROW_STD  = 311.76901171647364
# COL_MEAN = 719.5
# COL_STD = 415.69209358209673

def filter_annotations(box_annotations):
    annotations = []

    for i in range(len(box_annotations)):
        box_annotation = box_annotations[i]

        #skip ignored boxes
        if box_annotation["assoc_id"] == -1:
            continue

        annotations.append({
            "x0": box_annotation["x0"],
            "x1": box_annotation["x1"],
            "y0": box_annotation["y0"],
            "y1": box_annotation["y1"],
            "score": box_annotation["score"],
            "assoc_id": box_annotation["assoc_id"],
            "orig_index": i
        })

    
    return annotations

def get_boxes(annotations, image,
              resize, 
              file_id, 
              score_thresh,
              augment,
              ):
    
    out_centroids = []
    out_box_ims = []
    detection_indeces = []
    
    assoc_dict = {
        "matches": {},
        "non_matches": [],
        "unmatched": []
    }

    if augment:
        inds = np.random.choice(len(annotations), size=len(annotations), replace=False)
    else:
        inds = np.arange(len(annotations))

    for i in inds:
        annotation = annotations[i]
        #seg_inds = segmentations[i]

        x0 = annotation["x0"]
        y0 = annotation["y0"]
        x1 = annotation["x1"]
        y1 = annotation["y1"]
        score = annotation["score"]
        assoc_id = annotation["assoc_id"]
        det_id = annotation["orig_index"]

        #if score is less than score thresh drop
        #should not happen as using same score thresh but
        #adding in if I want to
        if score < score_thresh:
            continue

        #don't include if too small
        area = (x1-x0)*(y1-y0)
        if area == 0:
            continue

        crop_area = (int(x0), int(y0), int(x1), int(y1))
        box_im = image.crop(crop_area)
        box_im = resize(box_im)

        centroid = torch.as_tensor([(y1 + y0) / 2, (x1 + x0) / 2], dtype=torch.float32)

        out_box_ims.append(box_im)
        out_centroids.append(centroid)
        detection_indeces.append(det_id)

        if assoc_id >= 0:
            if assoc_id in assoc_dict["matches"]:
                raise RuntimeError('assoc_id appeared twice. debug: ' + file_id)    
            assoc_dict["matches"][assoc_id] = len(out_centroids) - 1
        elif assoc_dict == -2:
            assoc_dict["non_matches"].append(len(out_centroids) - 1)
        else:
            #raise RuntimeError('unmatched should not happen')
            assoc_dict["unmatched"].append(len(out_centroids) - 1)


    assoc_dict['num_dets'] = len(out_centroids)

    out_centroids = torch.vstack(out_centroids)
    out_box_ims = torch.stack(out_box_ims)
    detection_indeces = np.array(detection_indeces)

    return out_centroids, out_box_ims, detection_indeces, assoc_dict

def get_assoc_matrix(assoc_dict_0, assoc_dict_1, basename):
    num_dets_0 = assoc_dict_0['num_dets']
    num_dets_1 = assoc_dict_1['num_dets']

    match_matrix = np.zeros((num_dets_0 + 1, num_dets_1 + 1))

    matches_0 = assoc_dict_0['matches']
    matches_1 = assoc_dict_1['matches']

    if (not len(matches_0) == len(matches_1)):
        raise RuntimeError('Invalid match size, debug needed ' + basename)
    
    for assoc_id in matches_0:
        if not assoc_id in matches_1:
            raise RuntimeError('Mismatch assoc_id, debug needed: ' + basename)
            row = matches_0[assoc_id]
            match_matrix[row, -1] = 1.0
            continue
        
        row = matches_0[assoc_id]
        col = matches_1[assoc_id]

        match_matrix[row, col] = 1.0

    for row in assoc_dict_0['non_matches']:
        match_matrix[row, -1] = 1.0

    for col in assoc_dict_1['non_matches']:
        match_matrix[-1, col] = 1.0
    
    return match_matrix

def flip_image(image):
    hori_flippedImage = image.transpose(Image.FLIP_LEFT_RIGHT)
    return hori_flippedImage

def flip_annotations(annotations, width):
    for annotation in annotations:
        x0 = annotation["x0"]
        x1 = annotation["x1"]

        x1_new = width - 1 - x0
        x0_new = width - 1 - x1

        annotation["x0"] = x0_new
        annotation["x1"] = x1_new

def flip_data(image, annotations, width):
    image = flip_image(image)
    flip_annotations(annotations, width)

    return image, annotations

def rand_flip(image_0, image_1, annotations_0, annotations_1, width):
    rand_var = np.random.uniform()
    if rand_var < 0.5:
        image_0, annotations_0 = flip_data(image_0, annotations_0, width)
        image_1, annotations_1 = flip_data(image_1, annotations_1, width)

    return image_0, image_1, annotations_0, annotations_1

class AssociationDataset(Dataset):
    def __init__(self, annotatons_dir, 
                 images_dir, device, augment,
                 width=1440, height=1080, resize_size=64, score_thresh=0.4):
        
        self.annotation_paths = self.get_paths(annotatons_dir)
        self.images_dir = images_dir
        self.width = width
        self.height = height
        #TODO orig did not have antialis

        self.resize = T.Compose([
                      T.Resize((resize_size, resize_size), Image.BICUBIC),
                      T.ToTensor(),
                      T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        #self.resize = T.Resize((resize_size, resize_size), antialias=True)
        
        self.device = device
        self.augment = augment

        self.score_thresh = score_thresh

    def __len__(self):
        return len(self.annotation_paths)
    
    def __getitem__(self, idx):
        annotation_path = self.annotation_paths[idx]
        pair_annotations = read_dict(annotation_path)

        basename_0 = os.path.basename(pair_annotations['image_0']).replace('.png', '')
        basename_1 = os.path.basename(pair_annotations['image_1']).replace('.png', '')

        image_0_path = os.path.join(self.images_dir, basename_0 + '.png')
        image_1_path = os.path.join(self.images_dir, basename_1 + '.png')
        annotations_0 = pair_annotations['annotations_0']
        annotations_1 = pair_annotations['annotations_1']

        #rand flip
        if (self.augment) and (np.random.uniform() < 0.5):
            annotations_0, annotations_1 = annotations_1, annotations_0
            image_0_path, image_1_path = image_1_path, image_0_path

        #merge
        annotations_0 = filter_annotations(annotations_0)
        annotations_1 = filter_annotations(annotations_1)

        #image_0 = cv2.imread(image_0_path)
        #image_1 = cv2.imread(image_1_path)
        image_0 = Image.open(image_0_path).convert("RGB")
        image_1 = Image.open(image_0_path).convert("RGB")

        if self.augment:
            #random flip
            image_0, image_1, annotations_0, annotations_1 = rand_flip(image_0, image_1, annotations_0, annotations_1, self.width)

        box_centroids_0, box_ims_0, detection_indeces_0, assoc_dict_0 = get_boxes(annotations_0, image_0,                                                                                          
                                                                                  self.resize, '_'.join([basename_0, basename_1]), 
                                                                                  self.score_thresh,
                                                                                  self.augment)
        
        box_centroids_1, box_ims_1, detection_indeces_1, assoc_dict_1 = get_boxes(annotations_1, image_1, 
                                                                                  self.resize, '_'.join([basename_0, basename_1]), 
                                                                                  self.score_thresh,
                                                                                  self.augment)
        
        
        match_matrix = get_assoc_matrix(assoc_dict_0, assoc_dict_1, '_'.join([basename_0, basename_1]))
        match_matrix = torch.from_numpy(match_matrix).float()

        box_centroids = (box_centroids_0.to(self.device), box_centroids_1.to(self.device))
        box_ims = (box_ims_0.to(self.device), box_ims_1.to(self.device))
        gt_matrices = [match_matrix.to(self.device)]

        #no im when augment
        if self.augment:
            image_0_path = None
            image_1_path = None

        gt_vis = (detection_indeces_0, detection_indeces_1, image_0_path, image_1_path, os.path.basename(annotation_path))

        return box_centroids, box_ims, gt_matrices, gt_vis

    def get_paths(self, annotations_dir):
        paths = []
        for filename in os.listdir(annotations_dir):
            if not filename.endswith('.json'):
                continue

            paths.append(os.path.join(annotations_dir, filename))

        return paths

def collate_fn(data):
    zipped = zip(data)
    return list(zipped)

def get_data_loader(annotatons_dir, images_dir,
                    batch_size, shuffle, device,
                    augment):
    dataset = AssociationDataset(annotatons_dir, images_dir, device, augment)
    dloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return dloader
