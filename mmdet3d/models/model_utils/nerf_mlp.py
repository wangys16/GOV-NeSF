""" The MLPs and Voxels. """
import functools
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import (rearrange, reduce, repeat)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn



def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

# @torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var



class FusionNet(nn.Module):
    def __init__(self, in_feat_ch=32, n_samples=64, volume_dim=0, use_rde=True, **kwargs):
        super(FusionNet, self).__init__()
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples

        self.use_rde = use_rde
        self.ray_dir_fc = nn.Sequential(nn.Linear(3, 16),
                                        activation_func,
                                        nn.Linear(16, 32),
                                        activation_func)
        
        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*3+volume_dim, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+4*use_rde, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))
        
        self.view_transformer = LocalFeatureTransformer(d_model=32, 
                                    nhead=8, layer_names=['self'], attention='linear')
        
        self.ov_fc = nn.Sequential(nn.Linear(32, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

        self.viewToken = ViewTokenNetwork(dim=32)

    def posenc(self, d_hid, n_samples):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).cuda().float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, ray_diff, mask, globalfeat_=None, ov_feat=None, ray_dirs=None,):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        num_rays, num_samples, num_views = rgb_feat.shape[:3]
        rgb_in = rgb_feat[..., :3]
        
        weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]
        if globalfeat_ is not None:
            globalfeat = torch.cat([globalfeat, globalfeat_], dim=-1)


        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)

        x = rearrange(x, 'RN SN NV C -> (RN SN) NV C')
        if ray_dirs is not None:
            ray_dirs = rearrange(ray_dirs, 'RN SN NV C -> (RN SN) NV C')
        view_token = self.viewToken(x)
        view_token = rearrange(view_token, "B_RN_SN C -> B_RN_SN 1 C")
        x = torch.cat([view_token, x+self.ray_dir_fc(ray_dirs) if ray_dirs is not None else x], axis=1)
        x = self.view_transformer(x)

        x1 = rearrange(x, "B_RN_SN NV C -> NV B_RN_SN C")
        globalfeat = x1[0] #reference
        viewfeat = x1[1:]

        viewfeat = rearrange(viewfeat, "NV (RN SN) C -> RN SN NV C", RN=num_rays, SN=num_samples)
        globalfeat = rearrange(globalfeat, "(RN SN) C -> RN SN C", RN=num_rays, SN=num_samples)

        
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding[:, :globalfeat.shape[1]]
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        if self.use_rde:
            x_ = torch.cat([viewfeat, ray_diff], dim=-1)
        else:
            x_ = torch.cat([viewfeat], dim=-1)
        x_ = self.rgb_fc(x_)
        x_ = x_.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x_, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)

        x = self.ov_fc(viewfeat)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # ov blending
        ov_out = []
        length = math.ceil(x.shape[0] / 16)
        for i in range(16):
            ov_out.append(torch.sum(ov_feat[i*length:(i+1)*length]*blending_weights_valid[i*length:(i+1)*length], dim=2))
        ov_out = torch.cat(ov_out)
        return rgb_out, sigma_out, ov_out




from .grid_sample import grid_sample_2d, grid_sample_3d
from .attention.transformer import LocalFeatureTransformer

import math
PI = math.pi

class PositionEncoding(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L
        self.augmented = rearrange((PI * 2 ** torch.arange(-1, self.L - 1)), "L -> L 1 1 1")

    def forward(self, x):
        sin_term = torch.sin(self.augmented.type_as(x) * rearrange(x, "RN SN Dim -> 1 RN SN Dim")) # BUG? 
        cos_term = torch.cos(self.augmented.type_as(x) * rearrange(x, "RN SN Dim -> 1 RN SN Dim") )
        sin_cos_term = torch.stack([sin_term, cos_term])

        sin_cos_term = rearrange(sin_cos_term, "Num2 L RN SN Dim -> (RN SN) (L Num2 Dim)")

        return sin_cos_term


class RayTransformer(nn.Module):
    """
    Ray transformer
    """
    def __init__(self, args, img_feat_dim=32, fea_volume_dim=16, ov=False):
        super().__init__()

        self.args = args
        self.offset =  [[0, 0, 0]]

        self.volume_reso = args.volume_reso
        self.only_volume = False
        if self.only_volume:
            assert self.volume_reso > 0, "if only use volume feature, must have volume"

        self.img_feat_dim = img_feat_dim
        self.fea_volume_dim = fea_volume_dim if self.volume_reso > 0 else 0
        
        self.PE_d_hid = 8

        self.ov = ov

        # transformers
        self.density_view_transformer = LocalFeatureTransformer(d_model=self.img_feat_dim + self.fea_volume_dim, 
                                    nhead=8, layer_names=['self'], attention='linear')

        self.density_ray_transformer = LocalFeatureTransformer(d_model=self.img_feat_dim + self.PE_d_hid + self.fea_volume_dim, 
                                    nhead=8, layer_names=['self'], attention='linear')

        if self.only_volume:
            self.DensityMLP = nn.Sequential(
                nn.Linear(self.fea_volume_dim, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1))
        else:
            self.DensityMLP = nn.Sequential(
                nn.Linear(self.img_feat_dim + self.PE_d_hid + self.fea_volume_dim, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1))

        self.relu = nn.ReLU(inplace=True)

        # learnable view token
        self.viewToken = ViewTokenNetwork(dim=self.img_feat_dim + self.fea_volume_dim)
        self.softmax = nn.Softmax(dim=-2)

        # to calculate radiance weight
        self.linear_radianceweight_1_softmax = nn.Sequential(
            nn.Linear(self.img_feat_dim+3+self.fea_volume_dim, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 8), nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

        if ov:
            self.ov_weight_mlp = nn.Sequential(
            nn.Linear(self.img_feat_dim+self.fea_volume_dim, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 8), nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )


    def order_posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table)

        return sinusoid_table


    def forward(self, point3D, batch, source_imgs_feat, fea_volume=None, ov_feat=None):

        B, NV, _, H, W = batch['source_imgs'].shape
        _, RN, SN, _ = point3D.shape
        FDim = source_imgs_feat.size(2) # feature dim
        CN = len(self.offset)

        # calculate relative direction
        vector_1 = (point3D - repeat(batch['ref_pose_inv'][:,:3,-1], "B DimX -> B 1 1 DimX"))
        vector_1 = repeat(vector_1, "B RN SN DimX -> B 1 RN SN DimX")
        vector_2 = (point3D.unsqueeze(1) - repeat(batch['source_poses_inv'][:,:,:3,-1], "B L DimX -> B L 1 1 DimX")) # B L RN SN DimX
        vector_1 = vector_1/torch.linalg.norm(vector_1, dim=-1, keepdim=True) # normalize to get direction
        vector_2 = vector_2/torch.linalg.norm(vector_2, dim=-1, keepdim=True)
        dir_relative = vector_1 - vector_2 
        dir_relative = dir_relative.float()

        if self.args.volume_reso > 0: 
            assert fea_volume != None
            fea_volume_feat = grid_sample_3d(fea_volume, point3D.unsqueeze(1).float())
            fea_volume_feat = rearrange(fea_volume_feat, "B C RN SN -> (B RN SN) C")
        # -------- project points to feature map
        # B NV RN SN CN DimXYZ
        point3D = repeat(point3D, "B RN SN DimX -> B NV RN SN DimX", NV=NV).float()
        point3D = torch.cat([point3D, torch.ones_like(point3D[:,:,:,:,:1])], axis=4)
        
        # B NV 4 4 -> (B NV) 4 4
        points_in_pixel = torch.bmm(rearrange(batch['source_poses'], "B NV M_1 M_2 -> (B NV) M_1 M_2", M_1=4, M_2=4), 
                                rearrange(point3D, "B NV RN SN DimX -> (B NV) DimX (RN SN)"))
        
        points_in_pixel = rearrange(points_in_pixel, "(B NV) DimX (RN SN) -> B NV DimX RN SN", B=B, RN=RN)
        points_in_pixel = points_in_pixel[:,:,:3]
        # in 2D pixel coordinate
        mask_valid_depth = points_in_pixel[:,:,2]>0  #B NV RN SN
        mask_valid_depth = mask_valid_depth.float()
        points_in_pixel = points_in_pixel[:,:,:2] / points_in_pixel[:,:,2:3]

        img_feat_sampled, mask = grid_sample_2d(rearrange(source_imgs_feat, "B NV C H W -> (B NV) C H W"), 
                                rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"))
        img_rgb_sampled, _ = grid_sample_2d(rearrange(batch['source_imgs'], "B NV C H W -> (B NV) C H W"), 
                                rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2"))

        mask = rearrange(mask, "(B NV) RN SN -> B NV RN SN", B=B)
        mask = mask * mask_valid_depth
        img_feat_sampled = rearrange(img_feat_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)
        img_rgb_sampled = rearrange(img_rgb_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)

        # --------- run transformer to aggregate information
        # -- 1. view transformer
        x = rearrange(img_feat_sampled, "B NV C RN SN -> (B RN SN) NV C")
        
        if self.args.volume_reso > 0: 
            x_fea_volume_feat = repeat(fea_volume_feat, "B_RN_SN C -> B_RN_SN NV C", NV=NV)
            x = torch.cat([x, x_fea_volume_feat], axis=-1)

        # add additional view aggregation token
        view_token = self.viewToken(x)
        view_token = rearrange(view_token, "B_RN_SN C -> B_RN_SN 1 C")
        x = torch.cat([view_token, x], axis=1)
        x = self.density_view_transformer(x)

        x1 = rearrange(x, "B_RN_SN NV C -> NV B_RN_SN C")
        x = x1[0] #reference
        view_feature = x1[1:]

        if self.only_volume:
            x = rearrange(x_fea_volume_feat, "(B RN SN) NV C -> NV (B RN) SN C", B=B, RN=RN, SN=SN)[0]
        else:
            # -- 2. ray transformer
            # add positional encoding
            x = rearrange(x, "(B RN SN) C -> (B RN) SN C", RN=RN, B=B, SN=SN)
            x = torch.cat([x, repeat(self.order_posenc(d_hid=self.PE_d_hid, n_samples=SN).type_as(x), 
                                        "SN C -> B_RN SN C", B_RN = B*RN)], axis=2)
            x = self.density_ray_transformer(x)        

        srdf = self.DensityMLP(x)

        # calculate weight using view transformers result
        view_feature = rearrange(view_feature, "NV (B RN SN) C -> B RN SN NV C", B=B, RN=RN, SN=SN)
        dir_relative = rearrange(dir_relative, "B NV RN SN Dim3 -> B RN SN NV Dim3")

        x_weight = torch.cat([view_feature, dir_relative], axis=-1)
        x_weight = self.linear_radianceweight_1_softmax(x_weight)
        mask = rearrange(mask, "B NV RN SN -> B RN SN NV 1")
        x_weight[mask==0] = -1e9
        weight = self.softmax(x_weight)
        
        radiance = (img_rgb_sampled * rearrange(weight, "B RN SN L 1 -> B L 1 RN SN", B=B, RN=RN)).sum(axis=1)
        radiance = rearrange(radiance, "B DimRGB RN SN -> (B RN SN) DimRGB")

        return radiance, srdf, points_in_pixel  


class ViewTokenNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_parameter('view_token', nn.Parameter(torch.randn([1,dim])))

    def forward(self, x):
        return torch.ones([len(x), 1]).type_as(x) * self.view_token