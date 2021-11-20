import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from skimage import io
import skimage.transform
from collections import defaultdict


from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import boxes3d_kitti_camera_to_lidar, boxes3d_lidar_to_kitti_camera, boxes3d_kitti_camera_to_imageboxes
from lib.datasets.kitti.kitti_utils import random_flip_horizontal
from lib.datasets.kitti.kitti_utils import mask_boxes_outside_range_numpy
from lib.datasets.kitti.kitti_utils import get_pad_params
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from skimage import io

class KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg):
        # basic configuration
        self.root_dir = cfg.get('root_dir', '../../data/KITTI')
        self.split = split
        self.num_classes = len(cfg['writelist'])
        self.class_name = cfg['writelist']
        self.cls2id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(self.root_dir, 'object', 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        # data augmentation configuration
        self.istrain = True if split in ['train', 'trainval'] else False


        # others
        self.downsample = 4
        self.depth_downsample_factor = 4
        self.point_cloud_range = cfg['pc_range']


    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        image = io.imread(img_file)
        image = image[:, :, :3]  # Remove alpha channel
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx [str]: Index of the sample
        Returns:
            depth [np.ndarray(H, W)]: Depth map
        """
        depth_file = os.path.join(self.depth_dir, '%06d.png' % idx)
        assert os.path.exists(depth_file)
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        depth = skimage.transform.downscale_local_mean(image=depth,
                                                       factors=(self.depth_downsample_factor, self.depth_downsample_factor))

        return depth

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)


    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

        test_id = {'Car': 0, 'Pedestrian':1, 'Cyclist': 2}

        logger.info('==> Evaluating (official) ...')
        for category in self.class_name:
            results_str, results_dict = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            logger.info(results_str)


    def __len__(self):
        return self.idx_list.__len__()


    def __getitem__(self, item):
        input = {}
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        #index = 176

        img = self.get_image(index)
        if self.istrain==True:
            depth = self.get_depth_map(index)
        img_size = np.array(img.shape[:2], dtype=np.int32)

        calib = self.get_calib(index)
        if self.istrain==True:
            objects = self.get_label(index)

        V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
        R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
        V2R = R0 @ V2C
        trans_lidar_to_cam = V2R
        trans_cam_to_img = calib.P2

        if self.istrain==True:
            gt_names = []
            locations = []
            dims = []
            rotations_y = []
            bbox = []
            for object in objects:
                if object.cls_type not in self.class_name:
                    continue
                gt_names.append(object.cls_type)
                locations.append(object.pos)
                dims.append([object.l, object.h, object.w])
                rotations_y.append(object.ry)
                bbox.append(object.box2d)
            gt_names = np.array(gt_names)
            locations = np.array(locations)
            dims = np.array(dims)
            rotations_y = np.array(rotations_y)
            bbox = np.array(bbox)

            gt_boxes_camera = np.concatenate([locations, dims, rotations_y[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            ####### flig augmentation #########
            img, depth, gt_boxes_lidar, gt_bbox_2d = random_flip_horizontal(img, depth, gt_boxes_lidar, bbox, calib)

            gt_classes = np.array([self.cls2id[n] for n in gt_names], dtype=np.int32)
            gt_classes = gt_classes.reshape(-1, 1).astype(np.float32)

            gt_boxes_3d = np.concatenate((gt_boxes_lidar, gt_classes), axis=1)
            #gt_boxes_2d = np.concatenate((bbox, gt_classes), axis=1)

            ##### filter too far object #####
            mask = mask_boxes_outside_range_numpy(
                gt_boxes_3d, self.point_cloud_range, min_num_corners=1
            )
            gt_boxes_3d = gt_boxes_3d[mask]

            if len(gt_boxes_3d) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        if self.istrain==True:
            input['frame_id'] = index
            input['image_shape'] = img_size
            input['calib'] = calib
            input['gt_boxes'] = gt_boxes_3d
            input['gt_boxes2d'] = gt_bbox_2d
            input['images'] = img
            input['depth_maps'] = depth
            input['trans_lidar_to_cam'] = trans_lidar_to_cam
            input['trans_cam_to_img'] = trans_cam_to_img
        else:
            input['frame_id'] = index
            input['image_shape'] = img_size
            input['calib'] = calib
            input['images'] = img
            input['trans_lidar_to_cam'] = trans_lidar_to_cam
            input['trans_cam_to_img'] = trans_cam_to_img

        return input

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path +  '/' + ('%06d.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos


    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'root_dir': '../../../data/KITTI',
           'random_flip':0.0, 'random_crop':1.0, 'scale':0.8, 'shift':0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    dataset = KITTI_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break


    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
