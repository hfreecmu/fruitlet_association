import argparse
import os
import torch
from data.dataloader import get_data_loader
from utils.utils import load_checkpoint, read_model_params, plot_metrics
from utils.utils import extract_matches, evaluate_matches, vis_matches
from models.associator import Associator
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def infer(opt):
    #get dataloader
    dataloader = get_data_loader(opt.annotations_dir, opt.seg_dir,
                                 opt.images_dir, opt.seg_model_path,
                                 False, 1, False,
                                 opt.device)
    #

    #load model
    model_params = read_model_params(opt.params_path)
    model = Associator(model_params).to(opt.device)
    load_checkpoint(opt.checkpoint_epoch, opt.checkpoint_dir, model)
    model.eval()
    #

    mts = np.arange(1, 7)*0.1
    #TODO should these be averaged or total
    precisions = [None]*len(mts)
    recalls = [None]*len(mts)
    f1_scores = [None]*len(mts)

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            if not len(data) == 1:
                raise RuntimeError('Only batch size 1 supported in test')
            
            box_features, keypoint_vecs, is_tags, scores, gt_matrices, gt_vis = data[0][0]
            match_matrix = gt_matrices[0]

            match_scores = model(box_features, keypoint_vecs, is_tags, scores)

            for mt_ind in range(len(mts)):
                mt = mts[mt_ind]
                matches = extract_matches(match_scores, mt)
                precision, recall, f1_score = evaluate_matches(matches, match_matrix)
                if precisions[mt_ind] == None:
                    precisions[mt_ind] = []
                    recalls[mt_ind] = []
                    f1_scores[mt_ind] = []
                     
                precisions[mt_ind].append(precision)
                recalls[mt_ind].append(recall)
                f1_scores[mt_ind].append(f1_score)

                if mt == opt.vis_mt:
                    _, _, gt_centers_0, gt_centers_1, image_0_path, image_1_path, basename = gt_vis

                    output_path = os.path.join(opt.vis_dir, basename.replace('.json', '.png'))
                    vis_matches(matches, match_matrix, image_0_path, image_1_path, 
                                gt_centers_0, gt_centers_1, opt.vis_thickness, output_path)
            
    for mt_ind in range(len(mts)):
        precisions[mt_ind] = np.mean(precisions[mt_ind]) 
        recalls[mt_ind] = np.mean(recalls[mt_ind]) 
        f1_scores[mt_ind] = np.mean(f1_scores[mt_ind])    

    plot_metrics(opt.vis_dir, precisions, recalls, f1_scores, mts)

SEG_MODEL_PATH = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/segmentation/turk/mask_rcnn/mask_best.pth'
TRAIN_DIR = './datasets/train'
VAL_DIR = './datasets/val'
SEG_DIR = './preprocess_data/pair_segmentations'
IMAGES_DIR = './preprocess_data/pair_images'
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_dir', default=TRAIN_DIR)
    parser.add_argument('--images_dir', default=IMAGES_DIR)
    parser.add_argument('--seg_dir', default=SEG_DIR)
    parser.add_argument('--seg_model_path', default=SEG_MODEL_PATH)
    parser.add_argument('--params_path', default='./params/default_params.yml')

    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--checkpoint_epoch', type=int, required=True)

    parser.add_argument('--vis_dir', default='./vis')
    parser.add_argument('--vis_thickness', type=int, default=2)
    parser.add_argument('--vis_mt', type=float, default=0.4)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    infer(opt)