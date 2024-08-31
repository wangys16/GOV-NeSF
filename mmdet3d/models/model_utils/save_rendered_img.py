import logging
import os
import cv2
import imageio
import numpy as np
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm
import lpips

lpips_func = lpips.LPIPS(net="vgg").cuda()

def compute_psnr_from_mse(mse):
    return -10.0 * torch.log(mse) / np.log(10.0)

def compute_psnr(pred, target, mask=None):
    """Compute psnr value (we assume the maximum pixel value is 1)."""
    if mask is not None:
        pred, target = pred[mask], target[mask]
    mse = ((pred - target) ** 2).mean()
    return compute_psnr_from_mse(mse).cpu().numpy()

def compute_lpips(img1, img2):    
    lpips_func.requires_grad = False
    img1 = (img1 - 0.5) * 2
    img2 = (img2 - 0.5) * 2
    return lpips_func(img1.permute(2,0,1).type(torch.float16), img2.permute(2,0,1).type(torch.float16)).detach().cpu().numpy()


def compute_ssim(pred, target, mask=None):
    """Computes Masked SSIM following the neuralbody paper."""
    assert pred.shape == target.shape and pred.shape[-1] == 3
    if mask is not None:
        x, y, w, h = cv2.boundingRect(mask.cpu().numpy().astype(np.uint8))
        pred = pred[y : y + h, x : x + w]
        target = target[y : y + h, x : x + w]
    try:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), channel_axis=-1
        )
    except ValueError:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), multichannel=True
        )
    return ssim

def save_rendered_img(img_meta, rendered_results, work_dir):
    filename = img_meta[0]["filename"]
    scene = filename.split('/')[-2]
    metrics = dict()

    for ret in rendered_results:
        rgb = ret['outputs_coarse']['rgb']
        gt  = ret['gt_rgb']
        ov = ret['outputs_coarse']['ov']

    # # save images
    psnr_total = 0
    ssim_total = 0
    lpips_total = 0
    rsme = 0
    images = []
    for v in range(gt.shape[0]):
        img_to_save = torch.cat([rgb[v], gt[v]], dim=1)
        image_path = os.path.join(work_dir, 'visualization')
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        save_dir = os.path.join(image_path, scene+'.png')

        image = np.uint8(img_to_save.cpu().numpy() * 255.0)
        psnr = compute_psnr(rgb[v], gt[v], mask=None)
        psnr_total += psnr
        ssim = compute_ssim(rgb[v], gt[v], mask=None)
        ssim_total += ssim
        lpips = compute_lpips(rgb[v], gt[v])
        lpips_total += lpips

        images.append(image[:,:,::-1])
        cv2.imwrite(save_dir, image)


    return psnr_total/gt.shape[0], ssim_total/gt.shape[0], rsme/gt.shape[0], lpips_total/gt.shape[0], images



