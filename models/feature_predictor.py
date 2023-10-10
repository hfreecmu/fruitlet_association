import torch
import torch.nn as nn
from detectron2.modeling import build_model

class FeaturePredictor(nn.Module):
    def __init__(self, cfg):
        super(FeaturePredictor, self).__init__()

        self.cfg = cfg.clone()
        model = build_model(self.cfg)
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)['model'])

        self.preprocess_image = model.preprocess_image
        self.backbone = model.backbone
        self.box_in_features = model.roi_heads.box_in_features
        self.box_pooler = model.roi_heads.box_pooler

        self.mask_in_features = model.roi_heads.mask_in_features
        self.mask_pooler = model.roi_heads.mask_pooler
        self.mask_head = model.roi_heads.mask_head
            

    def forward(self, original_image, boxes):
        height, width = original_image.shape[:2]
        image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}

        images = self.preprocess_image([inputs])
        feature_dict = self.backbone(images.tensor)
        features = [feature_dict[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [boxes])

        mask_features = [feature_dict[f] for f in self.mask_in_features]
        mask_features = self.mask_pooler(mask_features, [boxes])
        pred_masks = self.mask_head(mask_features, [boxes], return_x=True)

        return box_features, pred_masks
