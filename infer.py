import argparse
import os
import torch
from data.dataloader import get_data_loader
from utils.utils import load_checkpoint, read_model_params
from utils.utils import extract_matches, evaluate_matches, vis_matches
from models.associator import Associator

import warnings
warnings.filterwarnings("ignore")


def infer(opt):
    #get dataloader
    dataloader = get_data_loader(opt.annotations_dir, opt.seg_dir,
                                 False, 1, False)
    #

    #load model
    model_params = read_model_params(opt.params_path)
    model = Associator(model_params).to(opt.device)
    load_checkpoint(opt.checkpoint_epoch, opt.checkpoint_dir, model)
    model.eval()
    #

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            if not len(data) == 1:
                raise RuntimeError('Only batch size 1 supported in test')
            
            box_features, keypoint_vecs, is_tags, scores, gt_matrices, gt_vis = data[0][0]
            match_matrix = gt_matrices[0]

            match_scores = model(box_features, keypoint_vecs, is_tags, scores)

            matches = extract_matches(match_scores, opt.match_thresh)
            #TODO plot these
            precision, recall, f1_score = evaluate_matches(matches, match_matrix)

            _, _, gt_centers_0, gt_centers_1, image_0_path, image_1_path, basename = gt_vis

            output_path = os.path.join(opt.vis_dir, basename.replace('.json', '.png'))
            vis_matches(matches, match_matrix, image_0_path, image_1_path, 
                        gt_centers_0, gt_centers_1, opt.vis_thickness, output_path)
            

default_image_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/fg_bg_images'
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_dir', required=True)
    parser.add_argument('--seg_dir', required=True)

    parser.add_argument('--params_path', default='./params/default_params.yml')

    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--checkpoint_epoch', type=int, required=True)

    parser.add_argument('--vis_dir', default='./vis')
    parser.add_argument('--image_dir', default=default_image_dir)
    parser.add_argument('--vis_thickness', type=int, default=2)

    parser.add_argument('--match_thresh', type=float, default=0.5)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    infer(opt)