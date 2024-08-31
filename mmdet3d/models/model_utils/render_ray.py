# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import math
import os
rng = np.random.RandomState(234)
# from tqdm import tqdm

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################
def volume_sampling(sample_pts, features, aabb):
    B, C, D, W, H = features.shape
    '''
    Actually here is hard code since per gpu only occupy one scene. hard_code B=1.
    can directly use point xyz instead of aabb size
    '''
    assert B == 1
    aabb = torch.Tensor(aabb).to(sample_pts.device)
    N_rays, N_samples, coords = sample_pts.shape
    sample_pts = sample_pts.view(1, N_rays*N_samples, 1, 1, 3).repeat(B, 1, 1, 1, 1)
    aabbSize = aabb[1] - aabb[0]
    invgridSize = 1.0/aabbSize * 2
    norm_pts = (sample_pts-aabb[0]) * invgridSize - 1
    sample_features = F.grid_sample(features, norm_pts, align_corners=True, padding_mode="border")
    # 1, C, 1, 1, N_rays*N_samples
    masks = ((norm_pts < 1) & (norm_pts > -1)).float().sum(dim=-1)
    masks = (masks.view(N_rays, N_samples) == 3) # x,y,z should be all in the volume.

    # TODO: return a mask represent whether the point is placed in volume.
    # TODO: Use border sampling, them mask filter.
    return sample_features.view(C, N_rays, N_samples).permute(1, 2, 0).contiguous(), masks

def _compute_projection(img_meta, src=False, dataset='scannet'):
    # [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
    projection = []
    src = 'src_' if src else ''
    views = len(img_meta['lidar2img'][src+'extrinsic'])
    intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:4, :4])
    ratio = img_meta['ori_shape'][0] / img_meta['img_shape'][0]
    
    # print(img_meta['lidar2img']['intrinsic'][:4, :4], img_meta['ori_shape'], img_meta['img_shape'])
    intrinsic[:2] /= ratio
    # print(intrinsic)
    intrinsic = intrinsic.unsqueeze(0).view(1, 16).repeat(views, 1)

    img_size = torch.Tensor(img_meta['img_shape'][:2]).to(intrinsic.device)
    img_size = img_size.unsqueeze(0).repeat(views, 1)
    # use predicted pitch and roll for SUNRGBDTotal test

    extrinsics = []
    for v in range(views):
        extrinsics.append(
            torch.Tensor(img_meta['lidar2img'][src+'extrinsic'][v]).to(intrinsic.device))
    extrinsic = torch.stack(extrinsics).view(views, 16)
    train_cameras = torch.cat([img_size, intrinsic, extrinsic], dim=-1)
    
    return train_cameras.unsqueeze(0)

def compute_mask_points(feature, mask):
    weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
    mean = torch.sum(feature * weight, dim=2, keepdim=True)
    # TODO: his would be a problem since non-valid point we assign var = 0!!!
    var = torch.sum((feature - mean)**2 , dim=2, keepdim=True)
    var = var / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)
    var = torch.exp(-var)

    return mean, var


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    '''

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1) # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)       # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)       # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i:i+1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)     # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]      # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1]-bins_g[:, :, 0])

    return samples


def sample_along_camera_ray(ray_o, ray_d, depth_range,
                            N_samples,
                            inv_uniform=False,
                            det=False):
    '''
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    '''
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0]
    far_depth_value = depth_range[1]
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])

    if inv_uniform:
        start = 1. / near_depth     # [N_rays,]
        step = (1. / far_depth - start) / (N_samples-1)
        inv_z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]
        z_vals = 1. / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples-1)
        z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]

    if not det:
        # get intervals between samples
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand   # [N_rays, N_samples]

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o       # [N_rays, N_samples, 3]
    return pts, z_vals


########################################################################################################################
# ray rendering of nerf
########################################################################################################################

def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    '''
    :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param ray_d: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    '''
    rgb = raw[:, :, :3]     # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]    # [N_rays, N_samples]

    sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)

    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

    alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

    weights = alpha * T     # [N_rays, N_samples]
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

    ov = raw[:, :, 4:]
    ov_map = torch.sum(weights.unsqueeze(2).detach() * ov, dim=1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

    if mask is not None:
        mask = mask.float().sum(dim=1) > 8  
    depth_map = torch.sum(weights * z_vals, dim=-1) / (torch.sum(weights, dim=-1) + 1e-8)
    depth_map = torch.clamp(depth_map, z_vals.min(), z_vals.max())

    ret = OrderedDict([('rgb', rgb_map),
                        ('depth', depth_map),
                        ('ov', ov_map),
                        ('weights', weights),                # used for importance sampling of fine samples
                        ('mask', mask),
                        ('alpha', alpha),
                        ('z_vals', z_vals),
                        ('transparency', T)
                        ])

    return ret


def render_rays_func(ray_o,
                     ray_d,
                     cam_idx,
                     features_2D,
                     img,
                     aabb,
                     near_far_range,
                     N_samples,
                     data_root,
                     img_meta=None,
                     projector=None,
                     inv_uniform=False,
                     det=False,
                     white_bkgd=False,
                     gt_rgb=None,
                     gt_ov=None,
                     fusion_volume=None,
                     coarse_net=None,
                     ov_images=None,
                     eval_3d=False,
                     eval_2d=False,
                     dataset='scannet',
                     use_de=True,):

    ret = {'outputs_coarse': None,
           'outputs_fine': None,
           'gt_rgb': gt_rgb}

    ret['gt_ov'] = gt_ov

    pts, z_vals = sample_along_camera_ray(ray_o=ray_o,
                                          ray_d=ray_d,
                                          depth_range=near_far_range,
                                          N_samples=N_samples,
                                          inv_uniform=inv_uniform,
                                          det=det)
    scene = img_meta['filename'].split('/')[-2]
    ret['scene'] = scene


    if eval_3d:
        path = f'{data_root}/{dataset}_instance_data'
        pts3d = np.load(f'{path}/{scene}_vert.npy')
        if dataset == 'scannet':
            matrix = np.load(f'{path}/{scene}_axis_align_matrix.npy')
            pts3d = np.concatenate([pts3d[:,:3], np.ones((pts3d.shape[0],1))], axis=1)
            pts3d = (pts3d @ matrix.T)[:,:3].astype(np.float32)
        else:
            pts3d = pts3d[:,:3].astype(np.float32)
        pts3d = torch.tensor(pts3d, device=pts.device).unsqueeze(1)
    
    N_rays, N_samples = pts.shape[:2]


    img = img.permute(0,2,3,1).unsqueeze(0)
    train_camera = _compute_projection(img_meta, dataset=dataset).to(img.device)
    query_camera = _compute_projection(img_meta, src=True, dataset=dataset).to(img.device)
    if not (eval_3d and (not eval_2d)):
        rgb_feat, mask, ray_diff, ray_dirs = projector.compute(pts, img, train_camera, features_2D, query_camera=query_camera, grid_sample=True)     
        ray_diff = torch.gather(ray_diff, dim=0, index=cam_idx.view(1,-1,1,1,1).repeat(1,1,ray_diff.shape[2],ray_diff.shape[3],ray_diff.shape[4])).squeeze(0)
        feat_dim = rgb_feat.shape[-1] - 3
        
        pixel_mask = mask[..., 0].sum(dim=2) > 1
        

    if eval_3d:
        rgb_feat3d, mask3d, ray_diff3d, ray_dirs3d = projector.compute(pts3d, img, train_camera, features_2D, query_camera=query_camera, grid_sample=True)     
        
        valid3d = (mask3d.sum(-2)>1).squeeze(-1).squeeze(-1)
        ret['valid3d'] = valid3d.detach().cpu().numpy()

    if not (eval_3d and (not eval_2d)):
        globalfeat_, inbound_masks = volume_sampling(pts, fusion_volume, aabb)
        ov_feat, _ = projector.compute(pts, img, train_camera, ov_images, grid_sample=True,rgb=False)
        rgb_pts, density_pts, ov_pts = coarse_net(rgb_feat[:,:,:,:(3+feat_dim)], ray_diff, mask, globalfeat_.unsqueeze(-2), ov_feat, \
                                                    ray_dirs=ray_dirs if use_de else None, )

        raw_coarse = torch.cat([rgb_pts, density_pts, ov_pts], dim=-1)

    # Evaluate 3D Segmentation
    else:
        feat_dim = rgb_feat3d.shape[-1] - 3
        torch.cuda.empty_cache()
        feat_3d = []
        feat_3d_avg = []
        length = 4096
        iter = math.ceil(pts3d.shape[0] / length)
        locs = []
        masks = []
        for i in range(iter):
            globalfeat_, inbound_masks = volume_sampling(pts3d[i*length:(i+1)*length], fusion_volume, aabb)
            
            ov_feat3d, mask_, loc = projector.compute(pts3d[i*length:(i+1)*length], img, train_camera, ov_images, grid_sample=True,rgb=False, return_loc=True)
            locs.append(loc)
            masks.append(mask_)
            _, _, ov_pts = coarse_net(rgb_feat3d[i*length:(i+1)*length,:,:,:(3+feat_dim)], ray_diff3d[0,i*length:(i+1)*length], mask3d[i*length:(i+1)*length],
                                                        globalfeat_.unsqueeze(-2), ov_feat3d, ray_dirs=ray_dirs3d[i*length:(i+1)*length] if use_de else None)

            weights_avg = torch.ones_like(ov_feat3d[:,:,:,:1], device=ov_feat3d.device)
            weights_avg = weights_avg.masked_fill(mask3d[i*length:(i+1)*length]==0, -1e9).detach().cpu()
            weights_avg = F.softmax(weights_avg, dim=2)
            
            ov_feat3d = ov_feat3d.detach().cpu()
            feat_3d.append(ov_pts.detach().cpu())
            feat_3d_avg.append(torch.sum(ov_feat3d*weights_avg, dim=2))
        
        feat_3d = torch.cat(feat_3d, dim=0)
        feat_3d_avg = torch.cat(feat_3d_avg, dim=0)
        ret['feat_3d'] = feat_3d
        ret['feat_3d_avg'] = feat_3d_avg
        
        del weights_avg, globalfeat_, ov_feat3d, feat_3d, feat_3d_avg, rgb_feat3d, mask3d, pts3d
        torch.cuda.empty_cache()
    

    if not (eval_3d and (not eval_2d)):
        ret['sigma'] = density_pts
        outputs_coarse = raw2outputs(raw_coarse, z_vals, pixel_mask,
                                    white_bkgd=white_bkgd)
        ret['outputs_coarse'] = outputs_coarse

    return ret

def render_rays(ray_batch,
                features_2D,
                img,
                aabb,
                near_far_range,
                N_samples,
                data_root,
                N_rand=4096,
                img_meta=None,
                projector=None,
                inv_uniform=False,
                det=False,
                is_train=True,
                white_bkgd=False,
                fusion_volume=None,
                coarse_net=None,
                ov_images=None,
                eval_3d=False,
                eval_2d=False,
                dataset='scannet',
                use_de=True,):
    '''
    :param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
    :param model:  {'net_coarse':  , 'net_fine': }
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param det: if True, will deterministicly sample depths
    :return: {'outputs_coarse': {}, 'outputs_fine': {}}
    Chenfeng: note that there is a risk that data augmentation is random origin
    not influence nerf, but influnence using nerf mlp to esimate volme density
    '''
    ray_o = ray_batch['ray_o']
    ray_d = ray_batch['ray_d']
    gt_rgb = ray_batch['gt_rgb']
    gt_ov = ray_batch['gt_ov']
    nerf_sizes = ray_batch['nerf_sizes']
    src_cameras = torch.tensor(img_meta['lidar2img']['src_extrinsic'], device=ray_d.device)
    ov_dim = 512

    if is_train:
        indices = torch.arange(ray_d.shape[1], device=ray_d.device)[:,None].repeat(1, ray_d.shape[2]).view(-1)
        ray_o = ray_o.view(-1, 3)
        ray_d = ray_d.view(-1, 3)
        gt_rgb = gt_rgb.view(-1, 3)
        gt_ov = gt_ov.view(-1, ov_dim)
        
        total_rays = ray_d.shape[0]
        select_inds = rng.choice(total_rays, size=(N_rand,), replace=False)
        cam_idx = indices[select_inds]
        ray_o = ray_o[select_inds]
        ray_d = ray_d[select_inds]
        gt_rgb = gt_rgb[select_inds]
        gt_ov = gt_ov[select_inds]

        rets = render_rays_func(ray_o,
                        ray_d,
                        cam_idx,
                        features_2D,
                        img,
                        aabb,
                        near_far_range,
                        N_samples,
                        data_root,
                        img_meta,
                        projector,
                        inv_uniform,
                        det,
                        white_bkgd,
                        gt_rgb,
                        gt_ov=gt_ov,
                        fusion_volume=fusion_volume,
                        coarse_net=coarse_net,
                        ov_images=ov_images,
                        dataset=dataset,
                        use_de=use_de,)

    else:
        nerf_size = nerf_sizes[0]
        indices = torch.arange(ray_d.shape[1], device=ray_d.device)[:,None].repeat(1, ray_d.shape[2]).view(-1)
        view_num = ray_o.shape[1]
        H = nerf_size[0][0]
        W = nerf_size[0][1]
        ray_o = ray_o.view(-1, 3)
        ray_d = ray_d.view(-1, 3)
        gt_rgb = gt_rgb.view(-1, 3)
        assert view_num*H*W == ray_o.shape[0]
        num_rays = ray_o.shape[0]
        results = []
        rgbs = []
        N_rand = 4096 if not(eval_3d and (not eval_2d)) else 1
        N_samples = N_samples if not(eval_3d and (not eval_2d)) else 1
        end = num_rays if not(eval_3d and (not eval_2d)) else N_rand
        for i in range(0, end, N_rand):
            ray_o_chunck = ray_o[i: i + N_rand, :]
            ray_d_chunck = ray_d[i: i + N_rand, :]
            cam_idx = indices[i: i + N_rand]

            ret = render_rays_func(ray_o_chunck,
                        ray_d_chunck,
                        cam_idx,
                        features_2D,
                        img,
                        aabb,
                        near_far_range,
                        N_samples,
                        data_root,
                        img_meta,
                        projector,
                        inv_uniform,
                        True,
                        white_bkgd,
                        gt_rgb,
                        fusion_volume=fusion_volume,
                        coarse_net=coarse_net,
                        ov_images=ov_images,
                        eval_3d=eval_3d*(i==0),
                        eval_2d=eval_2d*(i==0),
                        dataset=dataset,
                        use_de=use_de,)
            results.append(ret)
       
        rgbs= []
        depths = []
        ovs = []

        level = '_coarse'
        if results[0]['outputs'+level] != None:
            for i in range(len(results)):
                rgb = results[i]['outputs'+level]['rgb']
                rgbs.append(rgb)
                depth = results[i]['outputs'+level]['depth']
                depths.append(depth)
                ov = results[i]['outputs'+level]['ov']
                ovs.append(ov)


        outputs_coarse = None
        if not (eval_3d and (not eval_2d)):
            outputs_coarse = {'rgb': torch.cat(rgbs, dim=0).view(view_num, H, W, 3),
                    'depth': torch.cat(depths, dim=0).view(view_num, H, W, 1),
                    }
        rets = {'outputs_coarse': outputs_coarse,
                'gt_rgb': gt_rgb.view(view_num, H, W, 3),
        }
        if not (eval_3d and (not eval_2d)):
            rets['outputs_coarse']['ov'] = torch.cat(ovs, dim=0).view(view_num, H, W, ov_dim)
            rets['gt_ov'] = gt_ov.view(view_num, H, W, ov_dim)
        
        if eval_3d:
            rets['feat_3d'] = results[0]['feat_3d']
            rets['feat_3d_avg'] = results[0]['feat_3d_avg']
            rets['valid3d'] = results[0]['valid3d']
        rets['scene'] = results[0]['scene']

    return rets
