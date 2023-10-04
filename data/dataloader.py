import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
from utils.utils import read_dict, read_pickle, create_cfg, read_pickle, write_pickle
import cv2
from models.feature_predictor import FeaturePredictor
from detectron2.structures import Boxes
import torchvision.transforms.functional as F
import copy

#https://pupil-apriltags.readthedocs.io/en/stable/api.html
#https://github.com/pupil-labs/apriltags
from pupil_apriltags import Detector as AprilTagDetector

#TODO augment we are missing from original is rand_remove_assoc because we do not have that concept

SEG_MODEL_PATH='/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/segmentation/turk/mask_rcnn/mask_best.pth'
#DUMP_DIR='/home/frc-ag-3/harry_ws/fruitlet_2023/scripts/inhand/fruitlet_association/datasets/DUMMY'

def detect_aprilttag(detector, im_path, tag_id):
    im = cv2.imread(im_path)
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray_image)

    corners = None
    for result in results:
        if result.tag_id == tag_id:
            corners = result.corners
            break
    
    if corners is None:
        raise RuntimeError('No april tag detected in: ' + im_path)
    
    tag_image = np.zeros_like(gray_image)
    tag_image = cv2.fillPoly(tag_image, [corners.astype(np.int32)], color=(255))
    tag_seg_inds = np.argwhere(tag_image > 0)
    return tag_seg_inds, corners

def merge_annotations(box_annotations, box_segmentations, tag_corners, tag_seg_inds):
    annotations = []
    segmentations = []

    tag_x0 = np.min(tag_corners[:, 0])
    tag_x1 = np.max(tag_corners[:, 0])
    tag_y0 = np.min(tag_corners[:, 1])
    tag_y1 = np.max(tag_corners[:, 1])

    annotations.append({
        "x0": tag_x0,
        "x1": tag_x1,
        "y0": tag_y0,
        "y1": tag_y1,
        "is_tag": True,
        "score": 0.99,
        "assoc_id": -1,
        "orig_index": -1
    })

    segmentations.append(tag_seg_inds)

    if not len(box_annotations) == len(box_segmentations):
        raise RuntimeError('mismatch box annotations and box segmentations')

    for i in range(len(box_annotations)):
        box_annotation = box_annotations[i]
        box_seg_inds = box_segmentations[i]

        annotations.append({
            "x0": box_annotation["x0"],
            "x1": box_annotation["x1"],
            "y0": box_annotation["y0"],
            "y1": box_annotation["y1"],
            "is_tag": False,
            "score": box_annotation["score"],
            "assoc_id": box_annotation["assoc_id"],
            "orig_index": i
        })

        segmentations.append(box_seg_inds)
    
    return annotations, segmentations

def get_basenames(joint_basenmane):
    splits = joint_basenmane.replace('.json', '').split('_')

    basename_0 = '_'.join(splits[0:4])
    basename_1 = '_'.join(splits[4:8])

    return basename_0, basename_1

def get_boxes(annotations, segmentations, augment, 
              width, height, resize, file_id, 
              score_thresh,
              max_shift=5,
              drop_prob=[0.5, 0.1, 0.2], #prob of dropping any, if we drop min, if we drop max
              score_shift=0.03
              ):
    
    out_boxes = []
    is_tags = []
    keypoint_vecs = []
    detection_indeces = []
    scores = []
    
    assoc_dict = {
        "matches": {},
        "non_matches": [],
        "unmatched": []
    }
    gt_centers = []

    #get normalized rows and cols
    #TODO could be done once
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    rows = 2*(rows / rows.max()) - 1
    cols = 2*(cols / cols.max()) - 1

    #augment by shuffling inds
    if augment:
        inds = np.random.choice(len(annotations), size=len(annotations), replace=False)
    else:
        inds = np.arange(len(annotations))

    for i in inds:
        annotation = annotations[i]
        seg_inds = segmentations[i]

        x0 = int(np.round(annotation["x0"]))
        y0 = int(np.round(annotation["y0"]))
        x1 = int(np.round(annotation["x1"]))
        y1 = int(np.round(annotation["y1"]))
        is_tag = annotation["is_tag"]
        score = annotation["score"]
        assoc_id = annotation["assoc_id"]
        det_id = annotation["orig_index"]

        #if score is less than score thresh drop
        #should not happen as using same score thresh but
        #adding in if I want to
        if score < score_thresh:
            continue

        if augment:
            #first see if we drop
            #but don't drop tag
            if not is_tag and np.random.uniform() < drop_prob[0]:
                if np.random.uniform() < np.random.uniform(drop_prob[1], drop_prob[2]):
                    continue

            #second randomly adjust score if not tag
            if not is_tag:
                score += np.random.uniform(-score_shift, score_shift)
                if score < score_thresh:
                    score = score_thresh
                if score > 0.99:
                    score = 0.99

            #third augment by shifting boxes
            shifts = np.random.randint(-max_shift, max_shift + 1, size=(4,))
            x0 = x0 + shifts[0]
            x1 = x1 + shifts[1]
            y0 = y0 + shifts[2]
            y1 = y1 + shifts[3]

        #make sure xs and ys are in valid range
        x0 = np.max([x0, 0])
        x1 = np.min([x1, width - 1])
        y0 = np.max([y0, 0])
        y1 = np.min([y1, height - 1])

        seg_inds = seg_inds[seg_inds[:, 0] >= 0]
        seg_inds = seg_inds[seg_inds[:, 0] < height]
        seg_inds = seg_inds[seg_inds[:, 1] >= 0]
        seg_inds = seg_inds[seg_inds[:, 1] < width]

        if seg_inds.shape[0] == 0:
            continue

        #if cut off border then just drop it
        #with area check below
        if x1 < x0:
            x1 = x0

        if y1 < y0:
            y1 = y0

        #don't include if too small
        area = (x1-x0)*(y1-y0)
        if area == 0:
            continue
        
        #create a segmented image for just the box
        box_seg_im = np.zeros((height, width))
        box_seg_im[seg_inds[:, 0], seg_inds[:, 1]] = 1.0

        #create keypoint vector of
        #rows, columns, segmentations
        #TODO are y1 and x1 inclusive or exclusive throghout this?
        box_rows = rows[y0:y1, x0:x1]
        box_cols = cols[y0:y1, x0:x1]
        box_seg = box_seg_im[y0:y1, x0:x1]

        keypoint_vec = np.stack([box_rows, box_cols, box_seg])
        keypoint_vec = torch.from_numpy(keypoint_vec).float()
        keypoint_vec = resize(keypoint_vec)

        out_boxes.append([x0, y0, x1, y1])
        is_tags.append(is_tag)
        keypoint_vecs.append(keypoint_vec)
        detection_indeces.append(det_id)
        scores.append(score)
        gt_centers.append(((y0 + y1)/2, (x0 + x1)/2))

        if assoc_id >= 0:
            if assoc_id in assoc_dict["matches"]:
                raise RuntimeError('assoc_id appeared twice. debug: ' + file_id)
            
            assoc_dict["matches"][assoc_id] = len(is_tags) - 1
        elif assoc_dict == -2:
            assoc_dict["non_matches"].append(len(is_tags) - 1)
        else:
            assoc_dict["unmatched"].append(len(is_tags) - 1)

        if is_tag:
            if 'tag_ind' in assoc_dict:
                raise RuntimeError('Dup tag')

            assoc_dict['tag_ind'] = len(is_tags) - 1


    assoc_dict['num_dets'] = len(is_tags)

    out_boxes = torch.as_tensor(np.vstack(out_boxes), dtype=torch.float32)
    is_tags = torch.as_tensor(is_tags).float()
    keypoint_vecs = torch.stack(keypoint_vecs)
    scores = torch.as_tensor(scores).float()
    detection_indeces = np.array(detection_indeces)
    gt_centers = np.array(gt_centers)

    return out_boxes, is_tags, keypoint_vecs, scores, detection_indeces, gt_centers, assoc_dict

def get_feature_vecs(boxes, im, feature_predictor, device, feature_dict):    
    boxes = Boxes(boxes).to(device)
    with torch.no_grad():
        box_features, _ = feature_predictor(original_image=im, boxes=boxes)
        # if not im_path in feature_dict:
        #     box_features, features = feature_predictor(original_image=im, boxes=boxes)
        #     dump_path = os.path.join(DUMP_DIR, str(hash(im_path)) + '.pkl')
        #     write_pickle(dump_path, [f.to('cpu') for f in features])
        #     feature_dict[im_path] = dump_path
        # else:
        #     features = [f.to(device) for f in read_pickle(feature_dict[im_path])]
        #     box_features = feature_predictor.get_box_features(features=features, boxes=boxes)

    return box_features.to('cpu')

def get_assoc_matrix(assoc_dict_0, assoc_dict_1, basename):
    num_dets_0 = assoc_dict_0['num_dets']
    num_dets_1 = assoc_dict_1['num_dets']

    match_matrix = np.zeros((num_dets_0 + 1, num_dets_1 + 1))
    mask_matrix = np.ones((num_dets_0 + 1, num_dets_1 + 1))

    matches_0 = assoc_dict_0['matches']
    matches_1 = assoc_dict_1['matches']

    for row in assoc_dict_0['unmatched']:
        mask_matrix[row, :] = 0

    for col in assoc_dict_1['unmatched']:
        mask_matrix[:, col] = 0

    #taking this out for augment
    # if not len(matches_0) == len(matches_1):
    #     raise RuntimeError('Invalid match size, debug needed: ' + basename)
    
    for assoc_id in matches_0:
        if not assoc_id in matches_1:
            #taking this out for augment
            continue
            #raise RuntimeError('Mismatch assoc_id, debug needed: ' + basename)
        
        row = matches_0[assoc_id]
        col = matches_1[assoc_id]

        match_matrix[row, col] = 1.0
        mask_matrix[row, :] = 1.0
        mask_matrix[:, col] = 1.0

    for row in assoc_dict_0['non_matches']:
        match_matrix[row, -1] = 1.0
        mask_matrix[row, :] = 1.0

    for col in assoc_dict_1['non_matches']:
        match_matrix[-1, col] = 1.0
        mask_matrix[:, col] = 1.0
    
    #now do tag
    if 'tag_ind' in assoc_dict_0 and 'tag_ind' in assoc_dict_1:
        row_tag_ind = assoc_dict_0['tag_ind']
        col_tag_ind = assoc_dict_1['tag_ind']
        match_matrix[row_tag_ind, col_tag_ind] = 1.0
        mask_matrix[row, :] = 1.0
        mask_matrix[:, col] = 1.0
    elif 'tag_ind' in assoc_dict_0:
        row_tag_ind = assoc_dict_0['tag_ind']
        match_matrix[row_tag_ind, -1] = 1.0
        mask_matrix[row, :] = 1.0
    elif 'tag_ind' in assoc_dict_1:
        col_tag_ind = assoc_dict_1['tag_ind']
        match_matrix[-1, col_tag_ind] = 1.0
        mask_matrix[:, col] = 1.0


    return match_matrix, mask_matrix

def rand_flip(descs_0, descs_1, kpts_0, kpts_1):
    rand_var = np.random.uniform()
    if rand_var < 0.5:
        #box descriptors we fliplr
        descs_0 = torch.fliplr(descs_0)
        descs_1 = torch.fliplr(descs_1)

        #keypoints for rows unaffected

        #keypoints, for columns we normally would do width - x ...
        #BUT because cols are between -1 and 1, we just negate it
        kpts_0[:, 1, :, :] = -kpts_0[:, 1, :, :]
        kpts_1[:, 1, :, :] = -kpts_1[:, 1, :, :]

        #box segmentations are also flipped left and right
        kpts_0[:, 2:3] = torch.fliplr(kpts_0[:, 2:3])
        kpts_1[:, 2:3] = torch.fliplr(kpts_1[:, 2:3])

    return descs_0, descs_1, kpts_0, kpts_1

#warp 2D points using homography
def warp_points(points, H):
    points_homo = np.ones((points.shape[0], 3))
    points_homo[:, 0:2] = points
    perspective_points_homo = (H @ points_homo.T).T
    perspective_points = perspective_points_homo[:, 0:2] / perspective_points_homo[:, 2:]

    return perspective_points

class AssociationDataset(Dataset):
    def __init__(self, annotatons_dir, segmentations_dir, augment,
                 width=1440, height=1080, resize_size=64, score_thresh=0.4,
                 model_path=SEG_MODEL_PATH, device='cuda'):
        self.annotation_paths = self.get_paths(annotatons_dir)
        self.segmentations_dir = segmentations_dir
        self.augment = augment
        self.width = width
        self.height = height
        #TODO orig did not have antialis
        self.resize = T.Resize((resize_size, resize_size), antialias=True)

        self.apriltag_detector = AprilTagDetector()
        
        self.device = device

        cfg = create_cfg(model_path, score_thresh)
        self.feature_predictor = FeaturePredictor(cfg).to(self.device)
        self.feature_predictor.eval()

        self.score_thresh = score_thresh
        self.feature_dict = dict()

        self.random_affine = T.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.75, 0.9))
        self.random_brightness = T.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.05)
        self.perspective_distortion_scale = 0.4
        self.gauss_blur = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5))

    def __len__(self):
        return len(self.annotation_paths)
    
    def __getitem__(self, idx):
        annotation_path = self.annotation_paths[idx]
        pair_annotations = read_dict(annotation_path)

        image_0_path = pair_annotations['image_0']
        image_1_path = pair_annotations['image_1']
        annotations_0 = pair_annotations['annotations_0']
        annotations_1 = pair_annotations['annotations_1']

        basename_0, basename_1 = get_basenames(os.path.basename(annotation_path))
        seg_0_path = os.path.join(self.segmentations_dir, basename_0 + '.pkl')
        seg_1_path = os.path.join(self.segmentations_dir, basename_1 + '.pkl')
        segmentations_0 = read_pickle(seg_0_path)
        segmentations_1 = read_pickle(seg_1_path)

        tag_id_0 = int(basename_0.split('_')[0])
        tag_id_1 = int(basename_0.split('_')[0])
        if not tag_id_0 == tag_id_1:
            raise RuntimeError('tag_id mismatch')

        tag_seg_inds_0, corners_0 = detect_aprilttag(self.apriltag_detector, image_0_path, tag_id_0)
        tag_seg_inds_1, corners_1 = detect_aprilttag(self.apriltag_detector, image_1_path, tag_id_1)

        #merge
        annotations_0, segmentations_0 = merge_annotations(annotations_0, segmentations_0, corners_0, tag_seg_inds_0)
        annotations_1, segmentations_1 = merge_annotations(annotations_1, segmentations_1, corners_1, tag_seg_inds_1)

        image_0 = cv2.imread(image_0_path)
        image_1 = cv2.imread(image_1_path)

        if self.augment:
            if np.random.uniform() < 0.5:
                if np.random.uniform() < 0.5:
                    if np.random.uniform() < 0.5:
                        image_0, annotations_0, segmentations_0 = self.augment_affine(image_0, annotations_0, segmentations_0)
                    else:
                        image_1, annotations_1, segmentations_1 = self.augment_affine(image_1, annotations_1, segmentations_1)
                else:
                    if np.random.uniform() < 0.5:
                        image_0, annotations_0, segmentations_0 = self.augment_perspective(image_0, annotations_0, segmentations_0)
                    else:
                        image_1, annotations_1, segmentations_1 = self.augment_perspective(image_1, annotations_1, segmentations_1)

        boxes_0, is_tags_0, keypoint_vecs_0, scores_0, detection_indeces_0, gt_centers_0, assoc_dict_0 = get_boxes(annotations_0, segmentations_0, self.augment, 
                                                                                                         self.width, self.height, 
                                                                                                         self.resize, basename_0, self.score_thresh)
        boxes_1, is_tags_1, keypoint_vecs_1, scores_1, detection_indeces_1, gt_centers_1, assoc_dict_1 = get_boxes(annotations_1, segmentations_1, self.augment, 
                                                                                                         self.width, self.height, 
                                                                                                         self.resize, basename_1, self.score_thresh)
        box_features_0 = get_feature_vecs(boxes_0, image_0, self.feature_predictor, self.device, self.feature_dict) 
        box_features_1 = get_feature_vecs(boxes_1, image_1, self.feature_predictor, self.device, self.feature_dict) 

        match_matrix, _ = get_assoc_matrix(assoc_dict_0, assoc_dict_1, os.path.basename(annotation_path))
        match_matrix = torch.from_numpy(match_matrix).float()
        #mask_matrix = torch.from_numpy(mask_matrix).float()

        if self.augment:
            #random flip
            box_features_0, box_features_1, keypoint_vecs_0, keypoint_vecs_1 = rand_flip(box_features_0, box_features_1, keypoint_vecs_0, keypoint_vecs_1)

        box_features = (box_features_0.to(self.device), box_features_1.to(self.device))
        keypoint_vecs = (keypoint_vecs_0.to(self.device), keypoint_vecs_1.to(self.device))
        is_tags = (is_tags_0.to(self.device), is_tags_1.to(self.device))
        scores = (scores_0.to(self.device), scores_1.to(self.device))
        gt_matrices = [match_matrix.to(self.device)]

        #don't want loading if augmented / affined
        if self.augment:
            image_0_path = None
            image_1_path = None

        gt_vis = (detection_indeces_0, detection_indeces_1, gt_centers_0, gt_centers_1, image_0_path, image_1_path, os.path.basename(annotation_path))

        return box_features, keypoint_vecs, is_tags, scores, gt_matrices, gt_vis

    def get_paths(self, annotations_dir):
        paths = []
        for filename in os.listdir(annotations_dir):
            if not filename.endswith('.json'):
                continue

            paths.append(os.path.join(annotations_dir, filename))

        return paths
    
    def augment_affine(self, image, annotations, segmentations):
        torch_im = torch.from_numpy(image).permute(2, 0, 1)

        angle, translations, scale, shear = T.RandomAffine.get_params(self.random_affine.degrees, 
                                                                      self.random_affine.translate,
                                                                      self.random_affine.scale,
                                                                      self.random_affine.shear,
                                                                      (torch_im.shape[-1], torch_im.shape[-2]))
    
        center = [torch_im.shape[-1] * 0.5, torch_im.shape[-2] * 0.5]
        translations = list(translations)
        shear = list(shear)
        M = F._get_inverse_affine_matrix(center, angle, translations, scale, shear)
        M = np.array([[M[0], M[1], M[2]],
                      [M[3], M[4], M[5]],
                      [0, 0, 1.0]])
        
        #have to invert not sure why torch does this
        M = np.linalg.inv(M)

        warped_segmentations = []
        for seg_inds in segmentations:
            seg_inds = np.stack((seg_inds[:, 1], seg_inds[:, 0]), axis=1)

            affine_seg_inds = warp_points(seg_inds, M)
            affine_seg_inds = np.round(affine_seg_inds).astype(int)

            affine_seg_inds = np.stack((affine_seg_inds[:, 1], affine_seg_inds[:, 0]), axis=1)
            warped_segmentations.append(affine_seg_inds)

        warped_annotations = []
        for annotation in annotations:
            x0 = annotation["x0"]
            x1 = annotation["x1"]
            y0 = annotation["y0"]
            y1 = annotation["y1"]

            box_seg_inds = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
            box_seg_inds = warp_points(box_seg_inds, M)

            x0 = np.min(box_seg_inds[:, 0])
            x1 = np.max(box_seg_inds[:, 0])
            y0 = np.min(box_seg_inds[:, 1])
            y1 = np.max(box_seg_inds[:, 1])    

            warped_annotation = copy.deepcopy(annotation)
            warped_annotation["x0"] = x0  
            warped_annotation["x1"] = x1  
            warped_annotation["y0"] = y0  
            warped_annotation["y1"] = y1   
            warped_annotations.append(warped_annotation)

        torch_affine_img = F.affine(torch_im, angle, translations, scale, shear)
        affine_img = torch_affine_img.permute(1, 2, 0).numpy()

        return affine_img, warped_annotations, warped_segmentations
    
    def augment_perspective(self, image, annotations, segmentations):
        torch_im = torch.from_numpy(image).permute(2, 0, 1)
        
        start_points, end_points = T.RandomPerspective.get_params(torch_im.shape[-1], 
                                                                  torch_im.shape[-2], 
                                                                  self.perspective_distortion_scale)

        #not sure why torch does opposite direction when documentation
        #says otherwise but it does
        H = F._get_perspective_coeffs(end_points, start_points)

        H = np.array([[H[0], H[1], H[2]],
                      [H[3], H[4], H[5]],
                      [H[6], H[7], 1.0]])
        
        warped_segmentations = []
        for seg_inds in segmentations:
            seg_inds = np.stack((seg_inds[:, 1], seg_inds[:, 0]), axis=1)

            perspective_seg_inds = warp_points(seg_inds, H)
            perspective_seg_inds = np.round(perspective_seg_inds).astype(int)

            perspective_seg_inds = np.stack((perspective_seg_inds[:, 1], perspective_seg_inds[:, 0]), axis=1)
            warped_segmentations.append(perspective_seg_inds)
        
        
        warped_annotations = []
        for annotation in annotations:
            x0 = annotation["x0"]
            x1 = annotation["x1"]
            y0 = annotation["y0"]
            y1 = annotation["y1"]

            box_seg_inds = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
            box_seg_inds = warp_points(box_seg_inds, H)

            x0 = np.min(box_seg_inds[:, 0])
            x1 = np.max(box_seg_inds[:, 0])
            y0 = np.min(box_seg_inds[:, 1])
            y1 = np.max(box_seg_inds[:, 1])    

            warped_annotation = copy.deepcopy(annotation)
            warped_annotation["x0"] = x0  
            warped_annotation["x1"] = x1  
            warped_annotation["y0"] = y0  
            warped_annotation["y1"] = y1   
            warped_annotations.append(warped_annotation)

        torch_perspective_img = F.perspective(torch_im, start_points, 
                                              end_points)
        perspective_img = torch_perspective_img.permute(1, 2, 0).numpy()
        
        return perspective_img, warped_annotations, warped_segmentations

def collate_fn(data):
    zipped = zip(data)
    return list(zipped)

def get_data_loader(annotatons_dir, segmentations_dir, augment, 
                    batch_size, shuffle):
    dataset = AssociationDataset(annotatons_dir, segmentations_dir, augment)
    dloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return dloader
