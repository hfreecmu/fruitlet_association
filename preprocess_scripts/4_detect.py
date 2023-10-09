import os
import cv2
import torch
import distinctipy
import numpy as np
from inhand_utils import read_dict, write_dict, get_new_basename
from inhand_utils import write_pickle, load_seg_model

def detect(model, image_path):
    im = cv2.imread(image_path)
    im_boxes = []
    segmentations = []

    image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image}
    with torch.no_grad():
        outputs = model([inputs])[0]

    masks = outputs['instances'].get('pred_masks').to('cpu').numpy()
    boxes = outputs['instances'].get('pred_boxes').to('cpu')
    scores = outputs['instances'].get('scores').to('cpu').numpy()

    num = len(boxes)

    for i in range(num):
        x0, y0, x1, y1 = boxes[i].tensor.numpy()[0]
        seg_inds = np.argwhere(masks[i, :, :] > 0)
        score = scores[i]

        im_boxes.append([float(x0), float(y0), float(x1), float(y1), float(score)])
        segmentations.append(seg_inds)

    return im_boxes, segmentations


model_file = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/segmentation/turk/mask_rcnn/mask_best.pth'
json_path = '../preprocess_data/pairs.json'
image_dir = '../preprocess_data/pair_images'
det_output_path = '../preprocess_data/pair_detections.json'
seg_output_dir = '../preprocess_data/pair_segmentations'
score_thresh = 0.4
vis = False
#vis_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/debug_detections'
#vis_thickness = 2

model = load_seg_model(model_file, score_thresh)
pairs = read_dict(json_path)

image_paths = set()
for pair in pairs:
    for image_path in pair:
        if not image_path in image_paths:
            image_paths.add(image_path)

full_boxes = {}
for tmp_image_path in image_paths:
    new_basename = get_new_basename(tmp_image_path)

    image_path = os.path.join(image_dir, new_basename)

    boxes, segmentations = detect(model, image_path)

    full_boxes[os.path.basename(image_path)] = boxes

    seg_output_path = os.path.join(seg_output_dir, new_basename.replace('.png', '.pkl'))
    write_pickle(seg_output_path, segmentations)

    if vis:
        im = cv2.imread(image_path)
        num_boxes = len(boxes)
        colors = distinctipy.get_colors(num_boxes)

        for i in range(num_boxes):
            color = ([int(255*colors[i][0]), int(255*colors[i][1]), int(255*colors[i][2])])
            x0, y0, x1, y1, _ = boxes[i]

            seg_inds = segmentations[i]
            im[seg_inds[:, 0], seg_inds[:, 1]] = color

            cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), color, vis_thickness)
        
        vis_im_path = os.path.join(vis_dir, new_basename)
        cv2.imwrite(vis_im_path, im)

write_dict(det_output_path, full_boxes)