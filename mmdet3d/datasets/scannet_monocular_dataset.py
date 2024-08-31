import numpy as np
import os.path as osp
from collections import defaultdict
from mmdet.datasets import DATASETS

from .custom_3d import Custom3DDataset
from .scannet_dataset import ScanNetDataset
from mmdet3d.core.bbox import DepthInstance3DBoxes
from .dataset_wrappers import MultiViewMixin
import pickle


@DATASETS.register_module()
class ScanNetMultiViewDataset(MultiViewMixin, Custom3DDataset):
    CLASSES = ScanNetDataset.CLASSES

    def get_data_info(self, index):
        
        info = self.data_infos[index]
        input_dict = defaultdict(list)

        axis_align_matrix = info['annos']['axis_align_matrix'].astype(np.float32)

        for i in range(len(info['img_paths'])):
            img_filename = osp.join(self.data_root, info['img_paths'][i])
            input_dict['img_prefix'].append(None)
            input_dict['img_info'].append(dict(filename=img_filename))
            ovn_filename = osp.join(self.data_root, info['ovmap_paths'][i])
            input_dict['ovmap_info'].append(dict(filename=ovn_filename))
            extrinsic = np.linalg.inv(axis_align_matrix @ info['extrinsics'][i])
            input_dict['lidar2img'].append(extrinsic.astype(np.float32))
            if self.modality["use_ray"]:
                c2w = (axis_align_matrix @ info['extrinsics'][i]).astype(np.float32)
                input_dict['c2w'].append(c2w)
                input_dict['camrotc2w'].append(c2w[0:3, 0:3])
                input_dict['lightpos'].append(c2w[0:3, 3])

        input_dict = dict(input_dict)
        origin = np.array([.0, .0, .5])
        if not (self.dataset=='mp'):
            input_dict['lidar2img'] = dict(
                extrinsic=input_dict['lidar2img'],
                intrinsic=info['intrinsics'].astype(np.float32),
                origin=origin.astype(np.float32)
            )
        else:
            try:
                input_dict['lidar2img'] = dict(
                    extrinsic=input_dict['lidar2img'],
                    intrinsic=info['intrinsics'],
                    origin=origin.astype(np.float32)
                )
            except:
                exit(0)

        if self.modality["use_ray"]:
            input_dict['ray_info'] = dict(
            c2w=input_dict['c2w'],
            camrotc2w=input_dict['camrotc2w'],
            lightpos=input_dict['lightpos'],
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
                return None
        else:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0,), dtype=np.long)

        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        axis_align_matrix = info['annos']['axis_align_matrix'].astype(np.float32)
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            axis_align_matrix=axis_align_matrix,
            )
        return anns_results
