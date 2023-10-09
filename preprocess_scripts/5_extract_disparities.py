import sys
sys.path.append('/home/frc-ag-3/harry_ws/fruitlet/repos/RAFT-Stereo')
sys.path.append('/home/frc-ag-3/harry_ws/fruitlet/repos/RAFT-Stereo/core')

import os
import numpy as np
import torch
from PIL import Image

from raft_stereo import RAFTStereo
from utils.utils import InputPadder

from inhand_utils import RaftDummy, read_dict

DEVICE = 'cuda'

def get_fast_model_args(restore_ckpt):
    args = RaftDummy()
    args.restore_ckpt = restore_ckpt
    args.shared_backbone = True
    args.n_downsample = 3
    args.n_gru_layers = 2
    args.slow_fast_gru = True
    args.valid_iters = 7
    args.corr_implementation = 'reg_cuda'
    args.mixed_precision = True

    args.hidden_dims = [128]*3
    args.corr_levels = 4
    args.corr_radius = 4
    args.context_norm = 'batch'

    return args 

def get_irvc_model_args(restore_ckpt):
    args = RaftDummy()
    args.restore_ckpt = restore_ckpt
    args.context_norm = 'instance'

    args.shared_backbone = False
    args.n_downsample = 2
    args.n_gru_layers = 3
    args.slow_fast_gru = False
    args.valid_iters = 32
    args.corr_implementation = 'reg'
    args.mixed_precision = False
    args.hidden_dims = [128]*3
    args.corr_levels = 4
    args.corr_radius = 4

    return args  

def get_middle_model_args(restore_ckpt):
    args = RaftDummy()
    args.restore_ckpt = restore_ckpt
    args.corr_implementation = 'alt'
    args.mixed_precision = True

    args.shared_backbone = False
    args.n_downsample = 2
    args.n_gru_layers = 3
    args.slow_fast_gru = False
    args.valid_iters = 32
    args.hidden_dims = [128]*3
    args.corr_levels = 4
    args.corr_radius = 4
    args.context_norm = 'batch'

    return args 

def load_raft_model(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    return model

def load_raft_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def extract_disparity(model, left_im_path, right_im_path, valid_iters):
    image1 = load_raft_image(left_im_path)
    image2 = load_raft_image(right_im_path)

    with torch.no_grad():
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        _, flow_up = model(image1, image2, iters=valid_iters, test_mode=True)
        flow_up = padder.unpad(flow_up).squeeze()

    disparity = -flow_up.cpu().numpy().squeeze()
    return disparity  

image_dir = '../preprocess_data/pair_images'
json_path = '../preprocess_data/pair_detections.json'
output_dir = '../preprocess_data/pair_disparities'

left_str = 'LEFT'
right_str = 'RIGHT'

# model_file = '/home/frc-ag-3/harry_ws/viewpoint_planning/segment_exp/src/fruitlet_disparity/models/raftstereo-realtime.pth'
model_file = '/home/frc-ag-3/harry_ws/viewpoint_planning/segment_exp/src/fruitlet_disparity/models/iraftstereo_rvc.pth'
#model_file = '/home/frc-ag-3/harry_ws/viewpoint_planning/segment_exp/src/fruitlet_disparity/models/raftstereo-middlebury.pth'

# raft_args = get_fast_model_args(model_file)
raft_args = get_irvc_model_args(model_file)
#raft_args = get_middle_model_args(model_file)

#start
raft_model = load_raft_model(raft_args)
image_dict = read_dict(json_path)

for basename in image_dict:
    left_path = os.path.join(image_dir, basename)

    right_path = left_path.replace('LEFT', 'RIGHT')
    disparity = extract_disparity(raft_model, left_path, right_path, raft_args.valid_iters)

    disparity_output_path = os.path.join(output_dir, basename.replace('.png', '.npy'))

    np.save(disparity_output_path, disparity)

