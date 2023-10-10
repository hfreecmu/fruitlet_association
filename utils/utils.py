import json
import pickle
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import yaml
import cv2

from detectron2.config import get_cfg
from detectron2 import model_zoo

#read json
def read_dict(path):
    with open(path) as f:
        data = json.load(f)
    return data

#write json
def write_dict(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

#read pkl file
def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
#read yaml file
def read_yaml(path):
    with open(path, 'r') as f:
        yaml_to_read = yaml.safe_load(f)

    return yaml_to_read

#read_pickle
def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

#write_pickle
def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    
#positionally encode
def positional_encode(p, L):
    x = p[0]
    y = p[1]

    x_out = np.zeros((1 + 2*L, x.shape[0], x.shape[1]))
    y_out = np.zeros((1 + 2*L, y.shape[0], y.shape[1]))

    x_out[0] = x
    y_out[0] = y

    for k in range(L):
        ykcos = np.cos(2**k * np.pi * y)
        yksin = np.sin(2**k * np.pi * y)

        xkcos = np.cos(2**k * np.pi * x)
        xksin = np.sin(2**k * np.pi * x)

        y_out[2*k + 1] = ykcos
        y_out[2*k + 2] = yksin

        x_out[2*k + 1] = xkcos
        x_out[2*k + 2] = xksin

    enc_out = np.concatenate([x_out, y_out])
    return enc_out

#read model params
def read_model_params(params_path):
    params = read_yaml(params_path)
    return params

#save model checkpoint
def save_checkpoint(epoch, checkpoint_dir, model):
    model_path = os.path.join(checkpoint_dir, 'epoch_%d.pth' % epoch)
    torch.save(model.state_dict(), model_path)

#load model checkpoint
def load_checkpoint(epoch, checkpoint_dir, model):
    model_path = os.path.join(checkpoint_dir, 'epoch_%d.pth' % epoch)
    model.load_state_dict(torch.load(model_path))

#plot and save scores
def plot_and_save_scores(checkpoint_dir, epochs, precisions, recalls, f1_scores):
    epochs = np.array(epochs)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

    scores = np.vstack((epochs, precisions, recalls, f1_scores))
    scores_np_path = os.path.join(checkpoint_dir, 'scores.npy')
    np.save(scores_np_path, scores)

    scoress_plt_plath = os.path.join(checkpoint_dir, 'scores.png')

    plt.plot(epochs, precisions, 'b')
    plt.plot(epochs, recalls, 'r')
    plt.plot(epochs, f1_scores, 'g')
    plt.savefig(scoress_plt_plath)

    plt.clf()

#plot and save loss
def plot_and_save_loss(checkpoint_dir, epochs, losses, is_train):
    epochs = np.array(epochs)
    losses = np.array(losses)

    if is_train:
        losses_np_path = os.path.join(checkpoint_dir, 'train_losses.npy')
        losses_plt_plath = os.path.join(checkpoint_dir, 'train_losses.png')
    else:
        losses_np_path = os.path.join(checkpoint_dir, 'losses.npy')
        losses_plt_plath = os.path.join(checkpoint_dir, 'losses.png')

    np.save(losses_np_path, np.vstack((epochs, losses)))

    plt.plot(epochs, losses, 'r')
    plt.savefig(losses_plt_plath)

    plt.clf()

#get loss, assumes both inputs are same device
def get_loss(match_scores, match_matrix):
    M_loss = -match_matrix[0:-1, 0:-1]*match_scores[0:-1, 0:-1]
    M_loss = M_loss.sum()
    if M_loss > 0:
     M_loss = M_loss / match_matrix[0:-1, 0:-1].sum()

    I_loss = -match_matrix[:, -1]*match_scores[:, -1]
    I_loss = I_loss.sum()
    if I_loss > 0:
        I_loss = I_loss / match_matrix[:, -1].sum()

    J_loss = -match_matrix[-1, :]*match_scores[-1, :]
    J_loss = J_loss.sum()
    if J_loss > 0:
        J_loss = J_loss / match_matrix[-1, :].sum()
    
    assoc_loss = M_loss + I_loss + J_loss
    return assoc_loss

#arange_like for extract_matches
def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

#extract matches
def extract_matches(scores, match_threshold):
    scores = scores.unsqueeze(0)

    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > match_threshold)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

    return {
        'matches0': indices0.squeeze(0), # use -1 for invalid match
        'matches1': indices1.squeeze(0), # use -1 for invalid match
        'matching_scores0': mscores0.squeeze(0),
        'matching_scores1': mscores1.squeeze(0),
    }

#evaluate matches
def evaluate_matches(matches, match_matrix):
    indices0 = matches['matches0'].detach().cpu().numpy()
    indices1 = matches['matches1'].detach().cpu().numpy()

    match_matrix = match_matrix.detach().cpu().numpy()

    pos_match_inds = np.argwhere(match_matrix[0:-1, 0:-1] > 0)
    #neg_row_inds = np.argwhere(match_matrix[0:-1, -1] > 0)
    #neg_col_inds = np.argwhere(match_matrix[-1, 0:-1] > 0)

    #start with precision
    #which is tp matches / tp + fp matches
    pres_tp = 0
    pres_fp = 0
    for row in range(indices0.shape[0]):
        col = indices0[row]
        
        if col == -1:
            continue

        if indices1[col] != row:
            raise RuntimeError('Why here, should not happen')

        if match_matrix[row, col] > 0:
            pres_tp += 1
        else:
            pres_fp += 1

    if pres_tp == 0:
        precision = 0
    else:
        precision = pres_tp / (pres_tp + pres_fp) 


    #now do recall
    #which is tp matches over matches which should be tp but are not
    rec_tp = 0
    rec_fn = 0
    for row, col in pos_match_inds:
        if indices0[row] == col:
            if indices1[col] != row:
                raise RuntimeError('Why here, should not happen')
            
            rec_tp += 1
        else:
            rec_fn += 1

    if rec_tp == 0:
        recall = 0
    else:
        recall = rec_tp / (rec_tp + rec_fn)

    if precision == 0 or recall == 0:
        f1_score = 0
    else:
        f1_score = 2*precision*recall / (precision + recall)
    
    return precision, recall, f1_score

def vis_matches(matches, match_matrix, image_0_path, image_1_path, 
                gt_centers_0, gt_centers_1, vis_thickness, output_path):
    
    im_0 = cv2.imread(image_0_path)
    im_1 = cv2.imread(image_1_path)

    comb_im = np.zeros((im_0.shape[0], 2*im_0.shape[1], im_0.shape[2]))
    comb_im[:, 0:im_0.shape[1]] = im_0
    comb_im[:, im_0.shape[1]:] = im_1

    indices0 = matches['matches0'].detach().cpu().numpy()
    #indices1 = matches['matches1'].detach().cpu().numpy()

    match_matrix = match_matrix.detach().cpu().numpy()

    for row in range(indices0.shape[0]):
        if indices0[row] == -1:
            continue

        col = indices0[row]

        if match_matrix[row, col] == 1.0:
            color = [0, 255, 0]
        else:
            color = [0, 0, 255]

        y0, x0 = gt_centers_0[row]
        y1, x1 = gt_centers_1[col]
        x1 = x1 + im_0.shape[1]

        cv2.line(comb_im, (int(x0), int(y0)), (int(x1), int(y1)), color, vis_thickness)

    cv2.imwrite(output_path, comb_im)

def plot_metrics(output_dir, precisions, recalls, f1_scores, match_thresholds):
    comb_path = os.path.join(output_dir, 'metrics.png')
    comp_np_path = comb_path.replace('.png', '.npy')

    plt.plot(match_thresholds, precisions, 'b', label="precision")
    plt.plot(match_thresholds, recalls, 'r', label="recall")
    plt.plot(match_thresholds, f1_scores, 'g', label="f1 score")
    plt.legend(loc="lower left")
    plt.xlabel("Matching Thresholds")
    plt.xticks(np.arange(min(match_thresholds), max(match_thresholds), 0.1))
    plt.savefig(comb_path)

    plt.clf()

    comb_np = np.zeros((len(match_thresholds), 4))
    comb_np[:, 0] = match_thresholds
    comb_np[:, 1] = precisions
    comb_np[:, 2] = recalls
    comb_np[:, 3] = f1_scores
    np.save(comp_np_path, comb_np)

#create feature pred cfg
def create_cfg(model_file, score_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_file 
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.INPUT.MIN_SIZE_TEST = 1080
    cfg.INPUT.MAX_SIZE_TEST = 1440

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    return cfg