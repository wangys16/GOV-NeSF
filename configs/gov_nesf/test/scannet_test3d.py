eval_2d = False
eval_3d = True
class_names = [
    'toilet', 'bed', 'chair', 'sofa', 'dresser', 'table', 'cabinet',
    'bookshelf', 'pillow', 'sink', 'bathtub', 'refrigerator', 'desk',
    'nightstand', 'counter', 'door', 'curtain', 'box', 'lamp', 'bag'
]
use_de = True
use_rde = True
length_train = 0
length_val = 8
dataset = 'scannet'
dataset_type = 'ScanNetMultiViewDataset'
dataset_path = 'scannet'
data_root = f'/dataset/yswang/ovn_data/{dataset_path}/'
model = dict(
    type='GOV_NeSF',
    pretrained='torchvision://resnet50',
    data_root=data_root,
    backbone=dict(
        type='ResUNet',
        coarse_out_ch=64,
        fine_out_ch=32,
        coarse_only=True,),
    neck_3d=dict(
        type='ResidualUNetSE3D',
        in_channels=128,
        out_channels=64,
        n_levels=3,
        f_maps=[16, 32, 64],
        final_sigmoid=False),
    voxel_size=(6.4/40, 6.4/40, 2.56/16),
    n_voxels=(40, 40, 16),
    aabb=([-2.7, -2.7, -0.78], [3.7, 3.7, 1.78]),
    near_far_range=[0.2, 8.0],
    N_samples=64,
    N_rand=2048,
    use_nerf_mask=True,
    squeeze_scale=4,
    eval_2d=eval_2d,
    eval_3d=eval_3d,
    length_val=length_val,
    length_train=length_train,
    use_de=use_de,
    use_rde=use_rde,
    )
train_cfg = dict()
test_cfg = dict(nms_pre=1000, iou_thr=0.25, score_thr=0.004)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
input_modality = dict(
    use_image=True,
    use_lidar=False,
    use_ray=True)
train_collect_keys = ['img', 'gt_bboxes_3d', 'gt_labels_3d']
test_collect_keys = ['img']

if input_modality['use_ray']:
    ray_list = [
        'lightpos',
        'nerf_sizes',
        'raydirs',
        'gt_images',
        'denorm_images'
    ]
    ray_list.append('gt_ov_images')
    ray_list.append('ov_sizes')
    ray_list.append('ov_images')
    for key in ray_list:
        train_collect_keys.append(key)
        test_collect_keys.append(key)
    test_collect_keys.append('gt_sem_images')

multi_view_loss = True
train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=33,
        data_root = data_root,
        n_images_sup=50,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(320, 240), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(240, 320))
        ],
        mean = [123.675, 116.28, 103.53],
        std = [58.395, 57.12, 57.375],
        margin = 10,
        depth_range=[0.5, 5.5],
        loading='random',
        nerf_target_views=3,
        ovn_transforms=[
            dict(type='Resize', img_scale=(320, 240), keep_ratio=True),
            dict(type='Pad', size=(240, 320))
        ],
        ),
    dict(type='RandomShiftOrigin', std=(.7, .7, .0)), # this may lead to some issues in nerf.
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=train_collect_keys)
]

test_pipeline = [
    dict(
        type='MultiViewPipeline',
        n_images=100,
        data_root = data_root,
        n_images_sup=300,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(320, 240), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(240, 320))
        ],
        mean = [123.675, 116.28, 103.53],
        std = [58.395, 57.12, 57.375],
        margin = 10,
        depth_range=[0.5, 5.5],
        loading="random",
        nerf_target_views=1,
        ovn_transforms=[
            dict(type='Resize', img_scale=(320, 240), keep_ratio=True),
            dict(type='Pad', size=(240, 320))
        ],
        is_test = True,
        sem_transforms=[
            dict(type='Resize', img_scale=(320, 240), keep_ratio=True, interpolation='nearest'),
            dict(type='Pad', size=(240, 320))
        ],
        eval_3d = eval_3d,
        ),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=test_collect_keys)
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + f'scannet_infos_train_{length_train}.pkl',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            filter_empty_gt=True,
            box_type_3d='Depth',
            ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f'scannet_infos_val_{length_val}.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f'scannet_infos_val_{length_val}.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        )
)
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=1, decay_mult=1.0), 
                         coarse_net=dict(lr_mult=0.5, decay_mult=1.0)))
        )
optimizer_config = dict(grad_clip=dict(max_norm=35.0, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=1e-5, by_epoch=False)
total_epochs = 11
checkpoint_config = dict(interval=1, max_keep_ckpts=-1)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
