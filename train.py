import argparse
import torch
from data.dataloader import get_data_loader
from models.associator import Associator
import torch.optim as optim
from utils.utils import get_loss, extract_matches, evaluate_matches
from utils.utils import save_checkpoint, plot_and_save_loss, plot_and_save_scores
from utils.utils import read_model_params
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#TODO
#find better eval metrics. prec/recall/f1 may not be the best

#no cudnn memory issue
torch.backends.cudnn.enabled = False

def train(opt):
    #get dataloaders
    train_dataloader = get_data_loader(opt.annotations_dir, opt.seg_dir,
                                       opt.images_dir, opt.seg_model_path,
                                       True, opt.batch_size, opt.shuffle,
                                       opt.device)
    val_dataloader = get_data_loader(opt.val_dir, opt.seg_dir,
                                     opt.images_dir, opt.seg_model_path,
                                     False, 1, False, opt.device)
    #

    #load model
    model_params = read_model_params(opt.params_path)
    model = Associator(model_params).to(opt.device)
    model.train()
    #

    #get optimizer
    optimizer = optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    #optimizer = optim.Adam(model.parameters(), opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    #

    #for plotting
    loss_array = []
    plot_epochs = []

    precision_array = []
    recall_array = []
    f1_array = []
    val_loss_array = []
    #

    for epoch in range(opt.num_epochs):
        losses = []
        for _, data in enumerate(train_dataloader):
            loss = []
            optimizer.zero_grad()
            for data_num in range(len(data)):
                box_features, keypoint_vecs, is_tags, scores, gt_matrices, _ = data[data_num][0]

                match_matrix = gt_matrices[0]

                match_scores = model(box_features, keypoint_vecs, is_tags, scores)

                assoc_loss = get_loss(match_scores, match_matrix)

                loss.append(assoc_loss)

            loss = torch.mean(torch.stack(loss))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        scheduler.step()

        epoch_loss = np.mean(losses)
        print('loss for epoch', epoch, 'is: ', epoch_loss)

        log_epoch = epoch + 1
        if (log_epoch% opt.log_steps) == 0:
            #start by saving model and training losses
            save_checkpoint(epoch + 1, opt.checkpoint_dir, model)
            loss_array.append(epoch_loss)
            plot_epochs.append(log_epoch)
            plot_and_save_loss(opt.checkpoint_dir, plot_epochs, loss_array, True)

            #now evaluate
            model.eval()
            with torch.no_grad():
                precisions = []
                recalls = []
                f1_scores = []
                val_losses = []

                for _, data in enumerate(val_dataloader):
                    if not len(data) == 1:
                        raise RuntimeError('Only batch size 1 supported validate')
                    
                    box_features, keypoint_vecs, is_tags, scores, gt_matrices, _ = data[0][0]
                    match_matrix = gt_matrices[0]

                    match_scores = model(box_features, keypoint_vecs, is_tags, scores)

                    val_loss = get_loss(match_scores, match_matrix)

                    matches = extract_matches(match_scores, opt.match_thresh)
                    precision, recall, f1_score = evaluate_matches(matches, match_matrix)

                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1_score)
                    val_losses.append(val_loss.item())

            model.train()

            val_epoch_loss = np.mean(val_losses)
            print('val loss for epoch', epoch, 'is: ', val_epoch_loss)

            val_f1_score = np.mean(f1_scores)
            print('val f1 score for epoch', epoch, 'is: ', val_f1_score)

            precision_array.append(np.mean(precisions))
            recall_array.append(np.mean(recalls))
            f1_array.append(val_f1_score)
            val_loss_array.append(val_epoch_loss)

            plot_and_save_scores(opt.checkpoint_dir, plot_epochs, precision_array,
                                 recall_array, f1_array)
            plot_and_save_loss(opt.checkpoint_dir, plot_epochs, val_loss_array, False)



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
    parser.add_argument('--val_dir', default=VAL_DIR)
    parser.add_argument('--seg_model_path', default=SEG_MODEL_PATH)
    parser.add_argument('--params_path', default='./params/default_params.yml')


    parser.add_argument('--checkpoint_dir', default='./checkpoints')

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--milestones', type=list, default=[20, 40, 100])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--match_thresh', type=float, default=0.5)

    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    train(opt)
