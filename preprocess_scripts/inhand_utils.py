import os
import json
import yaml
import numpy as np
import cv2
import open3d
import pickle
import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model

class RaftDummy:
    def __init__(self) -> None:
        pass

def read_dict(path):
    with open(path) as f:
        data = json.load(f)
    return data

def write_dict(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def read_yaml(path):
    with open(path, 'r') as f:
        yaml_to_read = yaml.safe_load(f)

    return yaml_to_read

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def get_new_basename(image_path):
    color_dir = os.path.dirname(image_path)
    tag_dir = os.path.dirname(color_dir)
    tag_dirname = os.path.basename(tag_dir)

    tag_id_str, date_str = tag_dirname.split('_')
    year, month, day = date_str.split('-')[0:3]

    basename = os.path.basename(image_path)

    new_basename = '_'.join([tag_id_str, '-'.join([year, month, day]),
                            basename])
    
    return new_basename

def load_seg_model(model_file, score_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_file 
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.INPUT.MIN_SIZE_TEST = 1080
    cfg.INPUT.MAX_SIZE_TEST = 1440

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    model = build_model(cfg)
    model.load_state_dict(torch.load(model_file)['model'])
    model.eval()

    return model

def detect_apriltag(detector, gray_image, tag_id, 
                    estimate_tag_pose=False,
                    camera_params=None, #(fx, fy, cx, cy)
                    tag_size=0.02795):
    
    arpriltag_results = detector.detect(gray_image, 
                                        estimate_tag_pose=estimate_tag_pose,
                                        camera_params=camera_params,
                                        tag_size=tag_size)
    
    center = None
    tag_pos = None
    for result in arpriltag_results:
        if result.tag_id == tag_id:
            center = result.center
            if estimate_tag_pose:
                tag_pos = result.pose_t.reshape((3))
            break 

    if center is None:
        return False, None, None
    else:
        return True, center, tag_pos

def get_intrinsics(camera_info, include_fy=False):
    P = camera_info['P']
    f_norm = P[0]
    baseline = P[3] / P[0]
    cx = P[2]
    cy = P[6]

    if not include_fy:
        intrinsics = (baseline, f_norm, cx, cy)
    else:
        f_norm_y = P[5]
        intrinsics = (baseline, f_norm, cx, cy, f_norm_y)

    return intrinsics

def compute_points(disparity, intrinsics):
    baseline, f_norm, cx, cy = intrinsics
    stub = -baseline / disparity

    x_pts, y_pts = np.meshgrid(np.arange(disparity.shape[1]), np.arange(disparity.shape[0]))

    x = stub * (x_pts - cx)
    y = stub * (y_pts - cy)
    z = stub*f_norm

    points = np.stack((x, y, z), axis=2)

    return points

def create_point_cloud(cloud_path, points, colors):
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points)
    cloud.colors = open3d.utility.Vector3dVector(colors)

    open3d.io.write_point_cloud(
        cloud_path,
        cloud
    ) 

def bilateral_filter(disparity, intrinsics, 
                     bilateral_d=9, bilateral_sc=0.03, bilateral_ss=4.5):
    baseline, f_norm, _, _ = intrinsics
    stub = -baseline / disparity
    z = stub * f_norm
    z_new = cv2.bilateralFilter(z, bilateral_d, bilateral_sc, bilateral_ss)

    stub_new = z_new / f_norm
    disparity_new = -baseline / stub_new

    return disparity_new

def extract_depth_discontuinities(disparity, intrinsics,
                                  disc_use_rat=True,
                                  disc_rat_thresh=0.002,
                                  disc_dist_thresh=0.001):
    baseline, f_norm, _, _ = intrinsics
    stub = -baseline / disparity
    z = stub * f_norm

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(z, element)
    erosion = cv2.erode(z, element)

    dilation -= z
    erosion = z - erosion

    max_image = np.max((dilation, erosion), axis=0)

    if disc_use_rat:
        ratio_image = max_image / z
        _, discontinuity_map = cv2.threshold(ratio_image, disc_rat_thresh, 1.0, cv2.THRESH_BINARY)
    else:
        _, discontinuity_map = cv2.threshold(max_image, disc_dist_thresh, 1.0, cv2.THRESH_BINARY)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    discontinuity_map = cv2.morphologyEx(discontinuity_map, cv2.MORPH_CLOSE, element)

    return discontinuity_map

def extract_point_cloud(left_path, disparity_path, camera_info_path,
                        should_bilateral_filter, 
                        should_depth_discon_filter, 
                        should_distance_filter,
                        dist_thresh = 1.0):
    
    camera_info = read_yaml(camera_info_path)
    intrinsics = get_intrinsics(camera_info)

    disparity = np.load(disparity_path)
    inf_inds = np.where(disparity <= 0)

    disparity[inf_inds] = 1e-6

    im = cv2.imread(left_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if should_bilateral_filter:
        disparity = bilateral_filter(disparity, intrinsics)

    points = compute_points(disparity, intrinsics)
    colors = im.astype(float) / 255

    points[inf_inds] = np.nan
    colors[inf_inds] = np.nan

    if should_depth_discon_filter:
        discontinuity_map = extract_depth_discontuinities(disparity, intrinsics)
        discon_inds = np.where(discontinuity_map > 0) 
        points[discon_inds] = np.nan
        colors[discon_inds] = np.nan

    if should_distance_filter:
        dist_inds = np.where(np.linalg.norm(points, axis=2) > dist_thresh)
        points[dist_inds] = np.nan
        colors[dist_inds] = np.nan

    return points, colors
