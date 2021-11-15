import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
from easydict import EasyDict as edict

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg['anchorhead']
        input_channels = self.model_cfg['input_channels']
        num_class = len(model_cfg['class_names'])
        class_names = model_cfg['class_names']
        grid_size = np.array(model_cfg['grid_size'])
        point_cloud_range = np.array(model_cfg['pc_range'])
        predict_boxes_when_training = self.model_cfg['predict_boxes_when_training']

        super().__init__(
            model_cfg=self.model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg['use_direction_classifier']:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg['num_dir_bins'],
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict, istrain=True):
        spatial_features_2d = data_dict['spatial_features_2d'] #[B, 384, 188, 140]

        cls_preds = self.conv_cls(spatial_features_2d) #[B, 18, 188, 140]
        box_preds = self.conv_box(spatial_features_2d) #[B, 42, 188, 140]

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None: #default: None
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d) ##[B, 12, 188, 140]
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None


        if istrain:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            '''
            targets_dict:
            box_cls_labels: [4, 157920]
            box_reg_targets: [4, 157920, 7]
            reg_weights: [4, 157920]
            '''
            self.forward_ret_dict.update(targets_dict)

        if not istrain or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
