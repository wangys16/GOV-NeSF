import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import BasePoints, get_points_type
from .loading import LoadPointsFromFile
from mmdet.datasets.pipelines import Compose, RandomFlip, LoadImageFromFile
import mmcv
from .data_augment_utils import get_dtu_raydir
from PIL import Image
import cv2
import tifffile
import os
from .semantic_utils import PointSegClassMapping
import pandas as pd
from PIL import Image

@PIPELINES.register_module()
class MultiViewPipeline:
    def __init__(self,
        transforms,
        n_images,
        data_root,
        n_images_sup=50,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        margin=10,
        depth_range=[0.5, 5.5],
        loading='random',
        nerf_target_views=0,
        ovn_transforms=None,
        is_test=False,
        sem_transforms=None,
        eval_gt=True,
        blending=False,
        dataset='scannet',
        eval_3d=False,
        ):
        self.transforms = Compose(transforms)
        self.depth_transforms = Compose([transforms[1], transforms[3]])
        self.n_images = n_images
        self.data_root = data_root
        self.n_images_sup = n_images_sup
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.margin = margin
        self.depth_range = depth_range
        self.loading = loading
        self.sample_freq = n_images // (nerf_target_views + 1)
        self.nerf_target_views = nerf_target_views
        self.nerf_target_views = nerf_target_views
        self.ovn_transforms = Compose(ovn_transforms)
        self.is_test = is_test
        self.sem_transforms = Compose(sem_transforms) if is_test else None
        self.eval_gt = eval_gt
        self.blending = blending
        self.dataset = dataset
        self.eval_3d = eval_3d
        """
        For point-cloud below only
        """
        self.load_points = LoadPointsFromFile(
                coord_type='DEPTH',
                load_dim=6,
                use_dim=[0, 1, 2],
                shift_height=True)
        self.global_alignment = GlobalAlignment(rotation_axis=2)

        self.ov_dim = 512

        if is_test:
            path = data_root
            mapping_file = f'{path}/meta_data/scannetv2-labels.combined.tsv'
            mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
            scan_ids = mapping_file['id'].values
            nyu40_ids = mapping_file['nyu40id'].values
            scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
            for i in range(len(scan_ids)):
                scan2nyu[scan_ids[i]] = nyu40_ids[i]
            self.scan2nyu = scan2nyu
            self.label_mapping = PointSegClassMapping(
                valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
                max_cat_id=40
            )


    def __call__(self, results):
        imgs = []
        depths = []
        extrinsics = []
        src_extrinsics = []
        c2ws = []
        camrotc2ws = []
        lightposes = []
        pixels = []
        raydirs = []
        gt_images = []
        gt_depths = []
        denorm_imgs_list = []
        nerf_sizes = []
        gt_ov_images = []
        ov_sizes = []
        gt_sem_images = []
        ov_images = []
        if (self.loading == 'random') and not(self.is_test):
            ids = np.arange(len(results['img_info']))
            replace = True if self.n_images > len(ids) else False
            # ids = np.random.choice(ids, self.n_images, replace=replace)
            start = np.random.randint(0, max(len(ids)-self.n_images+1, 1))
            end = min(start + self.n_images, len(ids))
            ids = np.random.choice(np.arange(start, end), self.n_images, replace=replace)
            if self.nerf_target_views != 0:
                target_id = np.random.choice(
                    ids, self.nerf_target_views, replace=False)
                ids = np.setdiff1d(ids, target_id)
                ids = ids.tolist()
                target_id = target_id.tolist()
            # print(f'ids: {ids}, target_ids: {target_id}')

        elif self.is_test:
            ids = np.arange(len(results['img_info']))
            if not self.eval_3d:
                begin_id = np.random.randint(0, max(1, len(ids)-self.n_images+1))
                end_id = begin_id+self.n_images
            else:
                begin_id = 0
                end_id = len(ids)

            ids = np.sort(np.random.choice(ids[begin_id: end_id], min(self.n_images, len(ids)), replace=False))
            target_id = np.arange(self.sample_freq, self.sample_freq*(self.nerf_target_views+1), self.sample_freq)
            target_id = ids[target_id]
            if not self.eval_3d:
                ids = np.setdiff1d(ids, target_id).tolist()
            else:
                ids = ids.tolist()
            target_id = target_id.tolist()
            print(f'ids: {ids}, target_ids: {target_id}')

            
        else:
            ids = np.arange(len(results['img_info']))
            begin_id = 0
            ids = np.arange(
                begin_id, begin_id+self.n_images*self.sample_freq, self.sample_freq)
            if self.nerf_target_views != 0:
                target_id = ids

        if "pts_filename" in results.keys():
            results = self.load_points(results)
            results = self.global_alignment(results)

        ratio = 0
        for i in ids:
            _results = dict()
            for key in ['img_prefix', 'img_info']:
                _results[key] = results[key][i]
            _results = self.transforms(_results)
            ori_shape = _results['ori_shape']
            aft_shape = _results['img_shape']
            ratio = ori_shape[0] / aft_shape[0]

            denorm_img = mmcv.imdenormalize(
                    _results['img'], self.mean, self.std, to_bgr=True
                ).astype(np.uint8) / 255.0
            denorm_imgs_list.append(denorm_img)
            imgs.append(_results['img'])
            height, width = imgs[0].shape[:2]
            extrinsics.append(results['lidar2img']['extrinsic'][i])


            scene, ov_name = results['ovmap_info'][i]['filename'].split('/')[-2:]
            ovmap = tifffile.imread(f'{self.data_root}/posed_ovs/{scene}/{ov_name}')
                
            ovmap = ovmap.reshape(self.ov_dim, -1, ovmap.shape[-1]).transpose(1,2,0)
            ov_results = {}
            ov_results['filename'] = results['ovmap_info'][i]['filename']
            ov_results['ori_filename'] = results['ovmap_info'][i]['filename']
            ov_results['img'] = ovmap
            ov_results['img_shape'] = ovmap.shape
            ov_results['ori_shape'] = ovmap.shape
            ov_results['img_fields'] = ['img']

            ov_image = (np.float16(ovmap) -(2**15))/ (2**11)
            
            ov_images.append(ov_image)



        if "ray_info" in results.keys():
            if not (self.dataset=='mp'):
                intrinsics_nerf = results['lidar2img']['intrinsic'].copy()
                intrinsics_nerf[:2] = intrinsics_nerf[:2] / ratio
            assert self.nerf_target_views > 0
            for i in target_id:
                c2ws.append(results["c2w"][i])
                camrotc2ws.append(results["camrotc2w"][i])
                lightposes.append(results["lightpos"][i])
                px, py = np.meshgrid(
                    np.arange(self.margin, width - self.margin).astype(np.float32),
                    np.arange(self.margin, height- self.margin).astype(np.float32)
                )
                pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
                pixels.append(pixelcoords)
                if self.dataset == 'mp':
                    intrinsics_nerf = np.eye(4)
                    intrinsics_nerf[:3,:3] = results['lidar2img']['intrinsic'][i]
                    intrinsics_nerf[:2] = intrinsics_nerf[:2] / ratio
                raydir = get_dtu_raydir(
                    pixelcoords, intrinsics_nerf, results["camrotc2w"][i])
                raydirs.append(np.reshape(raydir.astype(np.float32), (-1, 3)))
                temp_results = dict()
                for key in ['img_prefix', 'img_info']:
                    temp_results[key] = results[key][i]
                temp_results_ = self.transforms(temp_results)
                denorm_imgs = mmcv.imdenormalize(
                    temp_results_['img'], self.mean, self.std, to_bgr=True
                ).astype(np.uint8)
                gt_rgb_shape = denorm_imgs.shape

                src_extrinsics.append(results['lidar2img']['extrinsic'][i])



                if not self.is_test or self.eval_gt:
                    scene, ov_name = results['ovmap_info'][i]['filename'].split('/')[-2:]
                    ovmap = tifffile.imread(f'{self.data_root}/posed_ovs/{scene}/{ov_name}')
                    

                    ovmap = ovmap.reshape(self.ov_dim, -1, ovmap.shape[-1]).transpose(1,2,0)
                    ov_image = (np.float16(ovmap) -(2**15))/ (2**11)
                    
                    gt_ov_image = np.reshape(ov_image[py.astype(np.int32), px.astype(np.int32), :], (-1, self.ov_dim))
                    ov_sizes.append(np.array(gt_ov_image.shape))
                    gt_ov_images.append(gt_ov_image)

                if self.is_test and not(self.dataset=='mp'):
                    sem_filename = results['ovmap_info'][i]['filename'].replace('image', 'semantic').replace('_ov.tiff', '.png')
                    semmap = Image.open(sem_filename)
                    semmap = np.asarray(semmap, dtype=np.int32)[:,:,None]
                    semmap = np.ascontiguousarray(semmap)
                    sem_results = {}
                    sem_results['filename'] = sem_filename
                    sem_results['ori_filename'] = sem_filename
                    sem_results['img'] = semmap
                    sem_results['img_shape'] = semmap.shape
                    sem_results['ori_shape'] = semmap.shape
                    sem_results['img_fields'] = ['img']
                    sem_results = self.sem_transforms(sem_results)
                    if self.dataset == 'scannet':
                        semmap = self.scan2nyu[sem_results['img']]
                        semmap = self.label_mapping(semmap)
                    else:
                        semmap = sem_results['img']
                    semmap = semmap[10:-10, 10:-10, None]
                    gt_sem_images.append(semmap)

                gt_image = denorm_imgs[py.astype(np.int32), px.astype(np.int32), :]
                nerf_sizes.append(np.array(gt_image.shape))
                gt_image = np.reshape(gt_image, (-1, 3))
                gt_images.append(gt_image/255.0)
                if "depth_info" in results.keys():
                    if '.npy' in results["depth_info"][i]["filename"]:
                        _results["depth"] = np.load(results["depth_info"][i]["filename"])
                    else:
                        depth_image = Image.open(results["depth_info"][i]["filename"].replace('images', 'depths'))
                        _results["depth"] = np.asarray(depth_image) / (1000*(1+3*(self.dataset=='mp')))
                        _results["depth"] = mmcv.imresize(_results["depth"], (gt_rgb_shape[1], gt_rgb_shape[0]))
                        
                    _results["depth"] = _results["depth"]
                    gt_depth = _results["depth"][py.astype(np.int32), px.astype(np.int32)]
                    gt_depths.append(gt_depth)


        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs

        if "ray_info" in results.keys():
            results['c2w'] = c2ws
            results['camrotc2w'] = camrotc2ws
            results['lightpos'] = lightposes
            results['pixels'] = pixels
            results['raydirs'] = raydirs
            results['gt_images'] = gt_images
            results['gt_depths'] = gt_depths
            results['nerf_sizes'] = nerf_sizes
            results['denorm_images'] = denorm_imgs_list
            results['lidar2img']['src_extrinsic'] = src_extrinsics

            results['ov_sizes'] = ov_sizes
            results['gt_ov_images'] = gt_ov_images
            results['ov_images'] = ov_images
            if self.is_test:
                results['gt_sem_images'] = gt_sem_images

            '''
            Hard code here!!!!!!!!! Should be carefully pick up the value.
            point-NeRF it also add middle points!!!
            One important idea here is that we sample more rays in the object bounding box
            as we already have bounding boxes!!! Take advantage of everything
            '''
            results['depth_range'] = np.array([self.depth_range])

        if len(depths) != 0:
            results['depth'] = depths
        results['lidar2img']['extrinsic'] = extrinsics
        return results


@PIPELINES.register_module()
class RandomShiftOrigin:
    def __init__(self, std):
        self.std = std

    def __call__(self, results):
        shift = np.random.normal(.0, self.std, 3)
        results['lidar2img']['origin'] += shift
        return results


@PIPELINES.register_module()
class KittiSetOrigin:
    def __init__(self, point_cloud_range):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.origin = (point_cloud_range[:3] + point_cloud_range[3:]) / 2.

    def __call__(self, results):
        results['lidar2img']['origin'] = self.origin.copy()
        return results


@PIPELINES.register_module()
class KittiRandomFlip:
    def __call__(self, results):
        if results['flip']:
            results['lidar2img']['intrinsic'][0, 2] = -results['lidar2img']['intrinsic'][0, 2] + \
                                                      results['ori_shape'][1]
            flip_matrix_0 = np.eye(4, dtype=np.float32)
            flip_matrix_0[0, 0] *= -1
            flip_matrix_1 = np.eye(4, dtype=np.float32)
            flip_matrix_1[1, 1] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0]
            extrinsic = flip_matrix_0 @ extrinsic @ flip_matrix_1.T
            results['lidar2img']['extrinsic'][0] = extrinsic
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 0], -center[:, 1]) - alpha
            center_flip = center
            center_flip[:, 1] *= -1
            alpha_flip = np.arctan2(center_flip[:, 0], -center_flip[:, 1]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results


@PIPELINES.register_module()
class SunRgbdSetOrigin:
    def __call__(self, results):
        intrinsic = results['lidar2img']['intrinsic'][:3, :3]
        extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
        projection = intrinsic @ extrinsic
        h, w, _ = results['ori_shape']
        center_2d_3 = np.array([w / 2, h / 2, 1], dtype=np.float32)
        center_2d_3 *= 3
        origin = np.linalg.inv(projection) @ center_2d_3
        results['lidar2img']['origin'] = origin
        return results


@PIPELINES.register_module()
class SunRgbdTotalLoadImageFromFile(LoadImageFromFile):
    def __call__(self, results):
        file_name = results['img_info']['filename']
        flip = file_name.endswith('_flip.jpg')
        if flip:
            results['img_info']['filename'] = file_name.replace('_flip.jpg', '.jpg')
        results = super().__call__(results)
        if flip:
            results['img'] = results['img'][:, ::-1]
        return results


@PIPELINES.register_module()
class SunRgbdRandomFlip:
    def __call__(self, results):
        if results['flip']:
            flip_matrix = np.eye(3)
            flip_matrix[0, 0] *= -1
            extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
            results['lidar2img']['extrinsic'][0][:3, :3] = flip_matrix @ extrinsic @ flip_matrix.T
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            center = boxes[:, :3]
            alpha = boxes[:, 6]
            phi = np.arctan2(center[:, 1], center[:, 0]) - alpha
            center_flip = center @ flip_matrix
            alpha_flip = np.arctan2(center_flip[:, 1], center_flip[:, 0]) + phi
            boxes_flip = np.concatenate([center_flip, boxes[:, 3:6], alpha_flip[:, None]], 1)
            results['gt_bboxes_3d'] = results['box_type_3d'](boxes_flip)
        return results


@PIPELINES.register_module()
class GlobalAlignment(object):
    """Apply global alignment to 3D scene points by rotation and translation.
    Args:
        rotation_axis (int): Rotation axis for points and bboxes rotation.
    Note:
        We do not record the applied rotation and translation as in
            GlobalRotScaleTrans. Because usually, we do not need to reverse
            the alignment step.
        For example, ScanNet 3D detection task uses aligned ground-truth
            bounding boxes for evaluation.
    """

    def __init__(self, rotation_axis):
        self.rotation_axis = rotation_axis

    def _trans_points(self, input_dict, trans_factor):
        """Private function to translate points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
            trans_factor (np.ndarray): Translation vector to be applied.
        Returns:
            dict: Results after translation, 'points' is updated in the dict.
        """
        input_dict['points'].translate(trans_factor)

    def _rot_points(self, input_dict, rot_mat):
        """Private function to rotate bounding boxes and points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
            rot_mat (np.ndarray): Rotation matrix to be applied.
        Returns:
            dict: Results after rotation, 'points' is updated in the dict.
        """
        # input should be rot_mat_T so I transpose it here
        input_dict['points'].rotate(rot_mat.T)

    def _check_rot_mat(self, rot_mat):
        """Check if rotation matrix is valid for self.rotation_axis.
        Args:
            rot_mat (np.ndarray): Rotation matrix to be applied.
        """
        is_valid = np.allclose(np.linalg.det(rot_mat), 1.0)
        valid_array = np.zeros(3)
        valid_array[self.rotation_axis] = 1.0
        is_valid &= (rot_mat[self.rotation_axis, :] == valid_array).all()
        is_valid &= (rot_mat[:, self.rotation_axis] == valid_array).all()
        assert is_valid, f'invalid rotation matrix {rot_mat}'

    def __call__(self, input_dict):
        """Call function to shuffle points.
        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after global alignment, 'points' and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        assert 'axis_align_matrix' in input_dict['ann_info'].keys(), \
            'axis_align_matrix is not provided in GlobalAlignment'

        axis_align_matrix = input_dict['ann_info']['axis_align_matrix']
        assert axis_align_matrix.shape == (4, 4), \
            f'invalid shape {axis_align_matrix.shape} for axis_align_matrix'
        rot_mat = axis_align_matrix[:3, :3]
        trans_vec = axis_align_matrix[:3, -1]

        self._check_rot_mat(rot_mat)
        self._rot_points(input_dict, rot_mat)
        self._trans_points(input_dict, trans_vec)

        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotation_axis={self.rotation_axis})'
        return repr_str
