from pathlib import Path
from sklearn.metrics import confusion_matrix

import torch
from skimage.io import imsave
from imgviz import label_colormap

from network.loss import Loss
from utils.base_utils import color_map_backward
from skimage.metrics import structural_similarity
import numpy as np
import os

from utils.draw_utils import concat_images_list

np.seterr(invalid='ignore')


def compute_psnr(img_gt, img_pr, use_vis_scores=False, vis_scores=None, vis_scores_thresh=1.5):
    if use_vis_scores:
        mask = vis_scores >= vis_scores_thresh
        mask = mask.flatten()
        img_gt = img_gt.reshape([-1, 3]).astype(np.float32)[mask]
        img_pr = img_pr.reshape([-1, 3]).astype(np.float32)[mask]
        mse = np.mean((img_gt - img_pr) ** 2, 0)

    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    mse = np.mean((img_gt - img_pr) ** 2, 0)
    mse = np.mean(mse)
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr


def nanmean(data, **args):
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)


class PSNR_SSIM(Loss):
    default_cfg = {
        'eval_margin_ratio': 1.0,
    }

    def __init__(self, cfg):
        super().__init__([])
        self.cfg = {**self.default_cfg, **cfg}

    def __call__(self, data_pr, data_gt, step, **kwargs):
        rgbs_gt = data_pr['pixel_colors_gt']  # 1,rn,3
        rgbs_pr = data_pr['pixel_colors_nr']  # 1,rn,3
        if 'que_imgs_info' in data_gt:
            h, w = data_gt['que_imgs_info']['imgs'].shape[2:]
        else:
            h, w = data_pr['que_imgs_info']['imgs'].shape[2:]
        rgbs_pr = rgbs_pr.reshape([h, w, 3]).detach().cpu().numpy()
        rgbs_pr = color_map_backward(rgbs_pr)

        rgbs_gt = rgbs_gt.reshape([h, w, 3]).detach().cpu().numpy()
        rgbs_gt = color_map_backward(rgbs_gt)

        h, w, _ = rgbs_gt.shape
        h_margin = int(h * (1 - self.cfg['eval_margin_ratio'])) // 2
        w_margin = int(w * (1 - self.cfg['eval_margin_ratio'])) // 2
        rgbs_gt = rgbs_gt[h_margin:h - h_margin, w_margin:w - w_margin]
        rgbs_pr = rgbs_pr[h_margin:h - h_margin, w_margin:w - w_margin]

        psnr = compute_psnr(rgbs_gt, rgbs_pr)
        ssim = structural_similarity(
            rgbs_gt, rgbs_pr, win_size=11, channel_axis=-1, data_range=255)
        outputs = {
            'psnr_nr': torch.tensor([psnr], dtype=torch.float32),
            'ssim_nr': torch.tensor([ssim], dtype=torch.float32),
        }

        def compute_psnr_prefix(suffix):
            if f'pixel_colors_{suffix}' in data_pr:
                rgbs_other = data_pr[f'pixel_colors_{suffix}']  # 1,rn,3
                # h, w = data_pr['shape']
                rgbs_other = rgbs_other.reshape(
                    [h, w, 3]).detach().cpu().numpy()
                rgbs_other = color_map_backward(rgbs_other)
                psnr = compute_psnr(rgbs_gt, rgbs_other)
                ssim = structural_similarity(
                    rgbs_gt, rgbs_other, win_size=11, channel_axis=-1, data_range=255)
                outputs[f'psnr_{suffix}'] = torch.tensor(
                    [psnr], dtype=torch.float32)
                outputs[f'ssim_{suffix}'] = torch.tensor(
                    [ssim], dtype=torch.float32)

        # compute_psnr_prefix('nr')
        compute_psnr_prefix('dr')
        compute_psnr_prefix('nr_fine')
        compute_psnr_prefix('dr_fine')
        return outputs


class VisualizeImage(Loss):
    def __init__(self, cfg):
        super().__init__([])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'que_imgs_info' in data_gt:
            h, w = data_gt['que_imgs_info']['imgs'].shape[2:]
        else:
            h, w = data_pr['que_imgs_info']['imgs'].shape[2:]

        def get_img(key):
            rgbs = data_pr[key]  # 1,rn,3
            rgbs = rgbs.reshape([h, w, 3]).detach().cpu().numpy()
            rgbs = color_map_backward(rgbs)
            return rgbs

        outputs = {}
        imgs = [get_img('pixel_colors_gt'), get_img('pixel_colors_nr')]
        if 'pixel_colors_dr' in data_pr:
            imgs.append(get_img('pixel_colors_dr'))
        if 'pixel_colors_nr_fine' in data_pr:
            imgs.append(get_img('pixel_colors_nr_fine'))
        if 'pixel_colors_dr_fine' in data_pr:
            imgs.append(get_img('pixel_colors_dr_fine'))

        data_index = kwargs['data_index']
        model_name = kwargs['model_name']
        Path(f'data/vis/{model_name}').mkdir(exist_ok=True, parents=True)
        if h <= 64 and w <= 64:
            imsave(
                f'data/vis/{model_name}/step-{step}-index-{data_index}.png', concat_images_list(*imgs))
        else:
            imsave(
                f'data/vis/{model_name}/step-{step}-index-{data_index}.jpg', concat_images_list(*imgs))
        return outputs
    
    
class VisualizeSemantic(Loss):
    def __init__(self, cfg):
        super().__init__([])
        self.num_classes = cfg['num_classes']
        self.work_dir = cfg['work_dir']
        if 'semantic_color_map' in cfg:
            self.color_map = torch.tensor(cfg['semantic_color_map'], dtype=torch.uint8)
        else:
            self.color_map = label_colormap(self.num_classes + 1)
            self.color_map = torch.tensor(self.color_map, dtype=torch.uint8)
    
    def __call__(self, results, gt_results, image, img_metas):
        filename = img_metas[0]["filename"]
        scene = filename.split('/')[-2]
        img_shape = results['shape'][:3]
        num_img = img_shape[0]
        pred = results['predicted_labels'].reshape(img_shape)
        gt = results['true_labels'].reshape(img_shape)
        pseudo = gt_results['predicted_labels'].reshape(img_shape)
        for idx in range(num_img):
            pred_, gt_, pseudo_ = self.color_map[pred[idx]], self.color_map[gt[idx]], self.color_map[pseudo[idx]]

            output = concat_images_list(*[image[idx], pred_, pseudo_, gt_])
            save_path = os.path.join(self.work_dir, 'visualization', scene)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            imsave(os.path.join(save_path, f'results_{idx}.png'), output)



# From https://github.com/Harry-Zhi/semantic_nerf/blob/a0113bb08dc6499187c7c48c3f784c2764b8abf1/SSR/training/training_utils.py
class IoU(Loss):
    default_cfg = {
        'ignore_label': 20,
        'num_classes': 20,
    }

    def __init__(self):
        super().__init__([])
        self.cfg = {**self.default_cfg}


    def __call__(self, rgb_preds, ray_batch, text_feat, eval_gt=False):
        true_labels = torch.cat(ray_batch['sem_gt_images']).squeeze(1).reshape(
            [-1]).detach().cpu().numpy()
        
        if not eval_gt:
            predict = rgb_preds[0]['outputs_coarse']['ov'].reshape(
                [-1,512])
        else:
            predict = rgb_preds[0]['gt_ov'].reshape(
                [-1,512])
        predict = predict / predict.norm(dim=-1, keepdim=True)
        predict = predict.detach().cpu().numpy()

        predicted_labels = (predict @ text_feat.T).argmax(-1)


        if self.cfg['ignore_label'] != -1:
            valid_pix_ids = true_labels != self.cfg['ignore_label']
        else:
            valid_pix_ids = np.ones_like(true_labels, dtype=bool)

        num_classes = self.cfg['num_classes']
        predicted_labels_ = predicted_labels[valid_pix_ids]
        true_labels_ = true_labels[valid_pix_ids]

        conf_mat = confusion_matrix(
            true_labels_, predicted_labels_, labels=list(range(num_classes)))
        norm_conf_mat = np.transpose(np.transpose(
            conf_mat) / conf_mat.astype(float).sum(axis=1))

        # missing class will have NaN at corresponding class
        missing_class_mask = np.isnan(norm_conf_mat.sum(1))
        exsiting_class_mask = ~ missing_class_mask

        class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
        total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
        ious = np.zeros(num_classes)
        for class_id in range(num_classes):
            ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
        miou = np.mean(ious[exsiting_class_mask])
        if np.isnan(miou):
            miou = 0.
            total_accuracy = 0.
            class_average_accuracy = 0.
        output = {
            'ious': ious,
            'existing_class_mask': exsiting_class_mask,
            'miou': torch.tensor([miou], dtype=torch.float32),
            'total_accuracy': torch.tensor([total_accuracy], dtype=torch.float32),
            'class_average_accuracy': torch.tensor([class_average_accuracy], dtype=torch.float32),
            'predicted_labels': predicted_labels,
            'true_labels': true_labels,
            'shape': torch.cat(ray_batch['sem_gt_images']).squeeze(1).shape,
        }
        return output


class IoU3D(Loss):
    default_cfg = {
        'ignore_label': 20,
        'num_classes': 20,
    }

    def __init__(self, data_root):
        super().__init__([])
        self.cfg = {**self.default_cfg}
        self.data_root = data_root
        self.ov_dim = 512
        self.mapping = np.zeros((41)) + 20
        valid_cat_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39)
        for i, idx in enumerate(valid_cat_ids):
            self.mapping[idx] = i


    def __call__(self, rgb_preds, ray_batch, text_feat, avg=False):
        avg = '' if not avg else '_avg'
        scene = rgb_preds[0]['scene']
        valid3d = rgb_preds[0]['valid3d']
        path = f'{self.data_root}/scannet_instance_data'
        true_labels = np.load(f'{path}/{scene}_sem_label.npy')
        true_labels = self.mapping[true_labels]
        
        predict = rgb_preds[0]['feat_3d'+avg].reshape([-1,self.ov_dim])
        predict = predict / predict.norm(dim=-1, keepdim=True)
        predict = predict.numpy()
        predicted_labels = (predict @ text_feat.T).argmax(-1)

        if self.cfg['ignore_label'] != -1:
            valid_pix_ids = (true_labels != self.cfg['ignore_label']) * valid3d
        else:
            valid_pix_ids = valid3d

        num_classes = self.cfg['num_classes']
        predicted_labels_ = predicted_labels[valid_pix_ids]
        true_labels_ = true_labels[valid_pix_ids]

        conf_mat = confusion_matrix(
            true_labels_, predicted_labels_, labels=list(range(num_classes)))
        norm_conf_mat = np.transpose(np.transpose(
            conf_mat) / conf_mat.astype(float).sum(axis=1))
        

        # missing class will have NaN at corresponding class
        missing_class_mask = np.isnan(norm_conf_mat.sum(1))
        exsiting_class_mask = ~ missing_class_mask

        class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
        total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
        ious = np.zeros(num_classes)
        for class_id in range(num_classes):
            ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
        miou = np.mean(ious[exsiting_class_mask])
        if np.isnan(miou):
            miou = 0.
            total_accuracy = 0.
            class_average_accuracy = 0.
        output = {
            'ious': ious,
            'existing_class_mask': exsiting_class_mask,
            'miou': torch.tensor([miou], dtype=torch.float32),
            'total_accuracy': torch.tensor([total_accuracy], dtype=torch.float32),
            'class_average_accuracy': torch.tensor([class_average_accuracy], dtype=torch.float32),
            'predicted_labels': predicted_labels,
            'true_labels': true_labels,
        }
        output['shape'] = torch.cat(ray_batch['sem_gt_images']).squeeze(1).shape
        return output

name2metrics = {
    'psnr_ssim': PSNR_SSIM,
    'vis_img': VisualizeImage,
    'vis_sem': VisualizeSemantic,
    'iou': IoU,
}


def psnr_nr(results):
    return np.mean(results['psnr_nr'])


def psnr_nr_fine(results):
    return np.mean(results['psnr_nr_fine'])

def miou(results):
    return np.mean(results['miou'])

name2key_metrics = {
    'psnr_nr': psnr_nr,
    'psnr_nr_fine': psnr_nr_fine,
    'miou': miou,
}
