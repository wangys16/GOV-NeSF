import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector
from ..model_utils.render_ray import render_rays
from ..model_utils.nerf_mlp import MultiHeadAttention, FusionNet
from ..model_utils.projection import Projector
from ..model_utils.save_rendered_img import save_rendered_img
from mmdet3d.core import bbox3d2result
import os
import numpy as np
from network.metrics import IoU, IoU3D, VisualizeSemantic
from prettytable import PrettyTable





@DETECTORS.register_module()
class GOV_NeSF(BaseDetector):
    def __init__(self,
                 backbone,
                 neck_3d,
                 n_voxels,
                 voxel_size,
                 data_root,
                 neck=None,
                 dataset='scannet',
                 head_2d=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 aabb=None,
                 near_far_range=None,
                 N_samples=40,
                 N_rand=4096,
                 use_nerf_mask=True,
                 nerf_mode="volume",
                 squeeze_scale=4,
                 rgb_supervision=True,
                 work_dir=None,
                 eval_2d=False,
                 eval_3d=False,
                 length_val=1,
                 length_train=0,
                 use_de=True,
                 use_rde=True,
                 ):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        self.neck_3d = build_neck(neck_3d)
        self.data_root = data_root
        self.head_2d = build_head(head_2d) if head_2d is not None else None
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.aabb=aabb
        self.near_far_range=near_far_range
        self.N_samples=N_samples
        self.N_rand=N_rand
        self.projector = Projector()
        self.squeeze_scale = squeeze_scale
        self.use_nerf_mask = use_nerf_mask
        self.rgb_supervision = rgb_supervision
        self.work_dir = work_dir
        nerf_feature_dim = 64
        self.eval_2d = eval_2d
        self.eval_3d = eval_3d
        self.logger = None
        self.dataset = dataset
        self.use_de = use_de
        self.use_rde = use_rde
        self.init_weights(pretrained=pretrained)
        self.length_val = int(length_val)
        self.length_train = int(length_train)
        color_map = [[174, 199, 232],  # wall
                    [152, 223, 138],  # floor
                    [31, 119, 180],   # cabinet
                    [255, 187, 120],  # bed
                    [188, 189, 34],   # chair
                    [140, 86, 75],    # sofa
                    [255, 152, 150],  # table
                    [214, 39, 40],    # door
                    [197, 176, 213],  # window
                    [148, 103, 189],  # bookshelf
                    [196, 156, 148],  # picture
                    [23, 190, 207],   # counter
                    [247, 182, 210],  # desk
                    [219, 219, 141],  # curtain
                    [255, 127, 14],   # refrigerator
                    [91, 163, 138],   # shower curtain
                    [44, 160, 44],    # toilet
                    [112, 128, 144],  # sink
                    [227, 119, 194],  # bathtub
                    [82, 84, 163],    # otherfurn
                    [0., 0., 0.],  # invalid
                    ]

        
        self.text_feat = torch.load('text_feat.pth').detach().cpu().numpy()
        self.len_class = 20
        self.class_names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtain', 'toilet',
                    'sink', 'bathtub', 'otherfurniture']
        self.semvis = VisualizeSemantic({'num_classes':self.len_class, 
                                         'work_dir':self.work_dir,
                                         'semantic_color_map':color_map})

        self.iou = IoU()
        self.iou3d = IoU3D(data_root=data_root)
        
        self.metric_keys = ['psnr', 'ssim', 'rmse', 'lpips', 'miou', 'total_accuracy', 'class_average_accuracy',
                            'gt_miou', 'gt_total_accuracy', 'gt_class_average_accuracy']
        self.metric3d_keys = ['miou', 'total_accuracy', 'class_average_accuracy']
        self.metrics = {}
        self.metrics3d = {}
        self.reset_val()
    
        self.coarse_net = FusionNet(in_feat_ch=32, n_samples=64, volume_dim=(neck_3d['out_channels']), \
                                        use_rde=self.use_rde)
        self.nerf_mlp = None
        self.semantic_mlp = None
        

        self.nerf_mode = nerf_mode

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)

        
        in_feat = neck["out_channels"] if self.neck is not None else 64
        self.mapping = nn.Sequential(
            nn.Linear(in_feat, nerf_feature_dim//2)
        )
        

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.N_samples)
    
    def posenc(self, d_hid, n_samples):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).cuda().float().unsqueeze(0)
        return sinusoid_table
    
    def reset_val(self):
        self.is_testing = 0
        self.len_val = 0
        if self.eval_2d:
            for key in self.metric_keys:
                self.metrics[key] = 0
            self.iou_map = np.array([0] * self.len_class, dtype=np.float64)
            self.gt_iou_map = np.array([0] * self.len_class, dtype=np.float64)
            self.iou_freq = np.array([0] * self.len_class)
        if self.eval_3d:
            for key in self.metric3d_keys:
                self.metrics3d[key] = 0
            self.iou3d_map = np.array([0] * self.len_class, dtype=np.float64)
            self.avg_iou3d_map = np.array([0] * self.len_class, dtype=np.float64)
            self.iou3d_freq = np.array([0] * self.len_class)
    
    def output_evaluation(self):
        self.is_testing = 0
        if self.eval_2d:
            tab = PrettyTable(self.metric_keys)
            row = []
            for key in self.metric_keys:
                row.append(np.float64(self.metrics[key]/self.len_val))
            tab.add_row(row)
            try:
                self.logger.info(tab)
            except Exception as e:
                print('e:', e)

            tab = PrettyTable(self.class_names)
            tab.add_row((self.iou_map/self.iou_freq).tolist())
            try:
                self.logger.info(tab)
            except Exception as e:
                print('e:', e)
            
            tab = PrettyTable(self.class_names)
            tab.add_row((self.gt_iou_map/self.iou_freq).tolist())
            try:
                self.logger.info(tab)
            except Exception as e:
                print('e:', e)

        if self.eval_3d:
            tab = PrettyTable(self.metric3d_keys)
            row = []
            for key in self.metric3d_keys:
                row.append(np.float64(self.metrics3d[key]/self.len_val))
            tab.add_row(row)
            try:
                self.logger.info(tab)
            except Exception as e:
                print('e:', e)

            tab = PrettyTable(self.class_names)
            tab.add_row((self.iou3d_map/self.iou3d_freq).tolist())
            try:
                self.logger.info(tab)
            except Exception as e:
                print('e:', e)
        
        if self.length_train == 0:
            exit(0)


    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        if self.neck is not None:
            self.backbone.init_weights(pretrained=pretrained)
            self.neck.init_weights()
        self.neck_3d.init_weights()
        if self.head_2d is not None:
            self.head_2d.init_weights()

        
    def extract_feat(self, img, img_metas, mode, depth=None, ray_batch=None):
        batch_size = img.shape[0]
        img = img.reshape([-1] + list(img.shape)[2:])
        if depth is not None:
            depth_bs = depth.shape[0]
            assert depth_bs == batch_size
            depth = depth.reshape([-1] + list(depth.shape)[2:])
        x = self.backbone(img)
        features_2d = self.head_2d.forward(x[-1], img_metas) if self.head_2d is not None else None
        x = self.neck(x)[0] if self.neck is not None else x
        x = x.reshape([batch_size, -1] + list(x.shape[1:]))

        stride = img.shape[-1] / x.shape[-1]
        assert stride == 4 
        stride = int(stride)

        volumes, volumes_mean, volumes_cov, valids = [], [], [], []
        rgb_preds = []
        densitys = []
        num = 0
        for feature, img_meta in zip(x, img_metas):
            angles = features_2d[0] if features_2d is not None and mode == 'test' else None 
            projection = self._compute_projection(img_meta, stride, angles, dataset=self.dataset).to(x.device)
            points = get_points(
                n_voxels=torch.tensor(self.n_voxels),
                voxel_size=torch.tensor(self.voxel_size),
                origin=torch.tensor(img_meta['lidar2img']['origin'])
            ).to(x.device)
            
            height = img_meta['img_shape'][0] // stride
            width = img_meta['img_shape'][1] // stride
            volume, valid = backproject(
                feature[:, :, :height, :width],
                points,
                projection,
                None,
                self.voxel_size)
            density = None
            volume_sum = volume.sum(dim=0)
            valid = valid.sum(dim=0)
            volume_mean = volume_sum / (valid + 1e-8)
            volume_mean[:, valid[0]==0] = .0

            volume_cov = torch.sum((volume - volume_mean.unsqueeze(0)) ** 2, dim=0) / (valid + 1e-8)
            volume_cov[:, valid[0]==0] = 1e6
            volume_cov = torch.exp(-volume_cov)

            volume_fusion = torch.cat([volume_mean, volume_cov], dim=0)

            volumes.append(volume_fusion)
            valids.append(valid)
        x = torch.stack(volumes)
        valids = torch.stack(valids)
        x = self.neck_3d(x)


        for volume_fusion, img_meta in zip(x, img_metas):
            n_channels, n_x_voxels, n_y_voxels, n_z_voxels = volume_mean.shape
            if ray_batch is not None:
                
                feature_2d = feature[:, :, :height, :width]
                n_v, C, height, width = feature_2d.shape
                feature_2d = feature_2d.contiguous().view(n_v, C, -1).permute(0, 2, 1)
                feature_2d = self.mapping(feature_2d
                    ).permute(0, 2, 1).contiguous().view(n_v, -1, height, width)

                denorm_images = ray_batch['denorm_images'][num:(num+1)]
                denorm_images = denorm_images.reshape([-1] + list(denorm_images.shape)[2:])
                if 'ov_images' in ray_batch:
                    ov_images = ray_batch['ov_images'][num:(num+1)]
                    ov_images = ov_images.reshape([-1] + list(ov_images.shape)[2:])
                else:
                    ov_images = None
                num += 1
                
                ret = render_rays(
                            ray_batch,
                            feature_2d,
                            denorm_images,
                            self.aabb,
                            self.near_far_range,
                            self.N_samples,
                            self.data_root,
                            self.N_rand,
                            img_meta,
                            self.projector,
                            is_train = mode == "train",
                            fusion_volume=volume_fusion.unsqueeze(0),
                            coarse_net=self.coarse_net,
                            ov_images=ov_images,
                            eval_3d=self.eval_3d,
                            eval_2d=self.eval_2d,
                            dataset=self.dataset,
                            use_de=self.use_de,
                            )
                rgb_preds.append(ret)


        return x, valids, features_2d, rgb_preds, densitys

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        
        ray_batchs = {}

        if "raydirs" in kwargs.keys():
            ray_batchs['ray_o'] = kwargs['lightpos']
            ray_batchs['ray_d'] = kwargs['raydirs']
            ray_batchs['gt_rgb'] = kwargs['gt_images']
            ray_batchs['gt_ov'] = kwargs['gt_ov_images']
            ray_batchs['nerf_sizes'] = kwargs['nerf_sizes']
            ray_batchs['denorm_images'] = kwargs['denorm_images']
            ray_batchs['ov_sizes'] = kwargs['ov_sizes']
            ray_batchs['ov_gt_images'] = kwargs['gt_ov_images']
            ray_batchs['ov_images'] = kwargs['ov_images']
            x, valids, features_2d, rgb_preds, densitys = self.extract_feat(
                    img, img_metas, 'train', ray_batch=ray_batchs, depth=None)

        else:
            x, valids, features_2d, rgb_preds, densitys = self.extract_feat(
                    img, img_metas, 'train')
        losses = {}
        if self.head_2d is not None:
            losses.update(self.head_2d.loss(*features_2d, img_metas))
        if len(ray_batchs) != 0:
            if self.rgb_supervision:
                losses.update(self.nvs_loss_func(rgb_preds))   
            losses.update(self.ov_loss_func(rgb_preds))     
        return losses

    def nvs_loss_func(self, rgb_pred):
        loss = 0
        for ret in rgb_pred:
            rgb = ret['outputs_coarse']['rgb']
            gt  = ret['gt_rgb']
            masks = ret['outputs_coarse']['mask']
            if self.use_nerf_mask:
                loss += torch.sum(
                    masks.unsqueeze(-1)*(rgb - gt)**2)/(masks.sum() + 1e-6)
            else:
                loss += torch.mean((rgb - gt)**2)
        results =  dict(loss_coarse_nvs=loss)
        return results

    def ov_loss_func(self, rgb_pred):
        loss = 0
        for ret in rgb_pred:
            level = 'coarse'
            ov = ret['outputs_'+level]['ov']
            gt  = ret['gt_ov']
            masks = ret['outputs_'+level]['mask']
            similarity_loss = 1-nn.CosineSimilarity()(ov, gt)
            if self.use_nerf_mask:
                loss += torch.sum(
                    masks*similarity_loss)/(masks.sum() + 1e-6)
            else:
                loss += torch.mean(similarity_loss)
        return dict(loss_ov=loss)

    def forward_test(self, img, img_metas, **kwargs):
        ray_batchs = {}
        if "raydirs" in kwargs.keys():
            ray_batchs['ray_o'] = kwargs['lightpos']
            ray_batchs['ray_d'] = kwargs['raydirs']
            ray_batchs['gt_rgb'] = kwargs['gt_images']
            ray_batchs['gt_ov'] = kwargs['gt_ov_images']
            ray_batchs['nerf_sizes'] = kwargs['nerf_sizes']
            ray_batchs['denorm_images'] = kwargs['denorm_images']
            ray_batchs['ov_sizes'] = kwargs['ov_sizes']
            ray_batchs['ov_gt_images'] = kwargs['gt_ov_images']
            ray_batchs['sem_gt_images'] = kwargs['gt_sem_images']
            ray_batchs['ov_images'] = kwargs['ov_images']
            
            return self.simple_test(
                img, img_metas, ray_batch=ray_batchs,
            )
        else:
            return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas, depth=None, ray_batch=None, evaluate_nerf=True):
        self.is_testing = 1
        
        x, valids, features_2d, rgb_preds, densitys = self.extract_feat(
            img, img_metas, 'test', depth, ray_batch)
            
        self.len_val += 1
            
        if evaluate_nerf and self.eval_2d:
            psnr, ssim, rmse, lpips, image = save_rendered_img(img_metas, rgb_preds, self.work_dir)
            self.metrics['psnr'] += psnr
            self.metrics['ssim'] += ssim
            self.metrics['rmse'] += rmse
            self.metrics['lpips'] += lpips
        
        torch.cuda.empty_cache()
        if self.eval_2d:
            results = self.iou(rgb_preds, ray_batch, self.text_feat)
            gt_results = self.iou(rgb_preds, ray_batch, self.text_feat, eval_gt=True)
            self.semvis(results, gt_results, image, img_metas)
            self.metrics['miou'] += results['miou']
            self.metrics['gt_miou'] += gt_results['miou']
            self.metrics['total_accuracy'] += results['total_accuracy']
            self.metrics['gt_total_accuracy'] += gt_results['total_accuracy']
            self.metrics['class_average_accuracy'] += results['class_average_accuracy']
            self.metrics['gt_class_average_accuracy'] += gt_results['class_average_accuracy']
            self.iou_freq[results['existing_class_mask']] += 1
            self.iou_map[results['existing_class_mask']] += results['ious'][results['existing_class_mask']]
            self.gt_iou_map[gt_results['existing_class_mask']] += gt_results['ious'][gt_results['existing_class_mask']]
            self.logger.info('scene: {}, psnr: {}, miou: {}, mAcc: {}, gt_miou: {}, gt_mAcc: {}'.format(rgb_preds[0]['scene'], psnr, \
                                            results['miou'], results['class_average_accuracy'], gt_results['miou'], gt_results['class_average_accuracy']))

        if self.eval_3d:
            results = self.iou3d(rgb_preds, ray_batch, self.text_feat)
            self.metrics3d['miou'] += results['miou']
            self.metrics3d['total_accuracy'] += results['total_accuracy']
            self.metrics3d['class_average_accuracy'] += results['class_average_accuracy']
            self.iou3d_freq[results['existing_class_mask']] += 1
            self.iou3d_map[results['existing_class_mask']] += results['ious'][results['existing_class_mask']]
            self.logger.info('scene: {}, miou3d: {}, mAcc3d: {}'.format(rgb_preds[0]['scene'], \
                                            results['miou'], results['class_average_accuracy']))
            

        if (self.len_val == self.length_val) and self.is_testing and (self.eval_2d or self.eval_3d):
            self.output_evaluation()
            self.reset_val()


        bbox_results = [{'boxes_3d': torch.tensor([],), 'scores_3d': torch.tensor([],), 'labels_3d': torch.tensor([],)} for _ in range(len(img_metas))]
        return bbox_results

    def aug_test(self, imgs, img_metas):
        pass

    def show_results(self, *args, **kwargs):
        pass

    @staticmethod
    def _compute_projection(img_meta, stride, angles, dataset='scannet'):
        projection = []
        if not dataset=='mp':
            intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
            ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
            intrinsic[:2] /= ratio
            if angles is not None:
                extrinsics = []
                for angle in angles:
                    extrinsics.append(get_extrinsics(angle).to(intrinsic.device))
            else:
                extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
            for extrinsic in extrinsics:
                projection.append(intrinsic @ extrinsic[:3])
        else:
            intrinsics = []
            ratio_0 = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
            ratio_1 = img_meta['ori_shape'][1] / (img_meta['img_shape'][1] / stride)
            for intrinsic in img_meta['lidar2img']['intrinsic']:
                intrinsic = intrinsic.copy()
                intrinsic[:1] /= ratio_0
                intrinsic[1:2] /= ratio_1
                intrinsics.append(torch.tensor(intrinsic).type(torch.float32))
            if angles is not None:
                extrinsics = []
                for angle in angles:
                    extrinsics.append(get_extrinsics(angle).to(intrinsic[0].device))
            else:
                extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
            for i, extrinsic in enumerate(extrinsics):
                projection.append(intrinsics[i] @ extrinsic[:3])
        return torch.stack(projection)

@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    # origin: point-cloud center.
    points = torch.stack(torch.meshgrid([
        torch.arange(n_voxels[0]), # 40 W width, x
        torch.arange(n_voxels[1]), # 40 D depth, y
        torch.arange(n_voxels[2]) # 16 H Heigh, z
    ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points

# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject(features, points, projection, depth, voxel_size):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)

    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
    ##### below is using depth to sample feature ########
    if depth is not None:
        depth = F.interpolate(depth.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=True).squeeze(1)
        for i in range(n_images):
            z_mask = z.clone() > 0
            z1 = z_mask[i, valid[i]].sum()
            z_mask[i, valid[i]] = (z[i, valid[i]] > depth[i, y[i, valid[i]], x[i, valid[i]]] - voxel_size[-1]) & \
                (z[i, valid[i]] < depth[i, y[i, valid[i]], x[i, valid[i]]] + voxel_size[-1])
            z2 = z_mask[i, valid[i]].sum()
            print('rate:', z2/z1)
            valid = valid & z_mask

    ######################################################
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)

    return volume, valid


# for SUNRGBDTotal test
def get_extrinsics(angles):
    yaw = angles.new_zeros(())
    pitch, roll = angles
    r = angles.new_zeros((3, 3))
    r[0, 0] = torch.cos(yaw) * torch.cos(pitch)
    r[0, 1] = torch.sin(yaw) * torch.sin(roll) - torch.cos(yaw) * torch.cos(roll) * torch.sin(pitch)
    r[0, 2] = torch.cos(roll) * torch.sin(yaw) + torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
    r[1, 0] = torch.sin(pitch)
    r[1, 1] = torch.cos(pitch) * torch.cos(roll)
    r[1, 2] = -torch.cos(pitch) * torch.sin(roll)
    r[2, 0] = -torch.cos(pitch) * torch.sin(yaw)
    r[2, 1] = torch.cos(yaw) * torch.sin(roll) + torch.cos(roll) * torch.sin(yaw) * torch.sin(pitch)
    r[2, 2] = torch.cos(yaw) * torch.cos(roll) - torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)

    # follow Total3DUnderstanding
    t = angles.new_tensor([[0., 0., 1.], [0., -1., 0.], [-1., 0., 0.]])
    r = t @ r.T
    # follow DepthInstance3DBoxes
    r = r[:, [2, 0, 1]]
    r[2] *= -1
    extrinsic = angles.new_zeros((4, 4))
    extrinsic[:3, :3] = r
    extrinsic[3, 3] = 1.
    return extrinsic
