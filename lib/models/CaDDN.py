import os
import cv2
import torch
import torch.nn as nn
import numpy as np
#
# from lib.backbones import dla
# from lib.backbones.dlaup import DLAUp
# from lib.backbones.hourglass import get_large_hourglass_net
# from lib.backbones.hourglass import load_pretrian_model
from lib.models.ffe.depth_ffe import DepthFFE
from lib.models.f2v.frustum_to_voxel import FrustumToVoxel
from lib.models.conv2d_collapse import Conv2DCollapse
from lib.models.base_bev_backbone import BaseBEVBackbone
from lib.models.dense_heads.anchor_head_single import AnchorHeadSingle
from easydict import EasyDict as edict

from lib.models.iou3d_nms import iou3d_nms_utils

class CaDDN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.depth_ffe = DepthFFE(self.cfg)
        self.frustum_to_voxel = FrustumToVoxel(self.cfg)
        self.conv2dcollapse = Conv2DCollapse(self.cfg)
        self.basebevbackbone = BaseBEVBackbone(self.cfg)
        self.anchorheadsingle = AnchorHeadSingle(self.cfg)

        self.num_class = self.cfg['class_names']

    def forward(self, input, istrain=True):
        input = self.depth_ffe(input, istrain)
        input = self.frustum_to_voxel(input, istrain)
        input = self.conv2dcollapse(input, istrain)
        input = self.basebevbackbone(input, istrain)
        input = self.anchorheadsingle(input, istrain)

        if istrain:
            loss, tb_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(input)
            return pred_dicts, recall_dicts


    def get_training_loss(self):
        loss_rpn, tb_dict_rpn = self.anchorheadsingle.get_loss()
        loss_depth, tb_dict_depth = self.depth_ffe.get_loss()

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_depth': loss_depth.item(),
            **tb_dict_rpn,
            **tb_dict_depth
        }

        loss = loss_rpn + loss_depth
        return loss, tb_dict

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = edict({'RECALL_THRESH_LIST': [0.3, 0.5, 0.7], 'SCORE_THRESH': 0.1, 'OUTPUT_RAW_SCORE': False, 'EVAL_METRIC': 'kitti', 'NMS_CONFIG': {'MULTI_CLASSES_NMS': False, 'NMS_TYPE': 'nms_gpu', 'NMS_THRESH': 0.01, 'NMS_PRE_MAXSIZE': 4096, 'NMS_POST_MAXSIZE': 500}})
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            assert batch_dict['batch_box_preds'].shape.__len__() == 3
            batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds


            cls_preds = batch_dict['batch_cls_preds'][batch_mask]

            src_cls_preds = cls_preds
            assert cls_preds.shape[1] in [1, len(self.num_class)]

            if not batch_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)



            cls_preds, label_preds = torch.max(cls_preds, dim=-1)

            label_preds = label_preds + 1
            selected, selected_scores = class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

            if post_process_cfg.OUTPUT_RAW_SCORE:
                max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                selected_scores = max_cls_preds[selected]

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]
    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


# if __name__ == '__main__':
#     import torch
#     net = CenterNet3D(backbone='dla34')
#     print(net)
#
#     input = torch.randn(4, 3, 384, 1280)
#     print(input.shape, input.dtype)
#     output = net(input)
#     print(output.keys())


