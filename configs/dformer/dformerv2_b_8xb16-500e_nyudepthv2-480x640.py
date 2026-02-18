# Auto-generated DFormer mmengine config (pure Jittor runtime).

dataset_name = 'NYUDepthv2'
data_root = 'datasets/NYUDepthv2'
num_classes = 40
image_height = 480
image_width = 640

train_batch_size = 16
val_batch_size = 8
num_train_imgs = 795

max_epochs = 500
eval_interval_epochs = 25
checkpoint_interval_epochs = 25
warmup_epochs = 10

iters_per_epoch = num_train_imgs // train_batch_size + 1
max_iters = max_epochs * iters_per_epoch
val_interval = eval_interval_epochs * iters_per_epoch
checkpoint_interval = checkpoint_interval_epochs * iters_per_epoch
warmup_iters = warmup_epochs * iters_per_epoch

train_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

model = dict(
    type='DFormerLegacySegmentor',
    backbone='DFormerv2_B',
    decoder='ham',
    decoder_embed_dim=512,
    num_classes=num_classes,
    drop_path_rate=0.2,
    aux_rate=0.0,
    pretrained_model='checkpoints/pretrained/DFormerv2_Base_pretrained.pth',
    syncbn=True,
)

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=8,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True, seed=12345),
    dataset=dict(
        type='RGBXSegDataset',
        dataset_name=dataset_name,
        data_root=data_root,
        split='train',
        backbone='DFormerv2_B',
        image_height=image_height,
        image_width=image_width,
        train_scale_array=train_scale_array,
        gt_transform=True,
        x_is_single_channel=True,
        file_length=train_batch_size * iters_per_epoch,
    ))

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=0,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RGBXSegDataset',
        dataset_name=dataset_name,
        data_root=data_root,
        split='val',
        test_mode=True,
        backbone='DFormerv2_B',
        image_height=image_height,
        image_width=image_width,
        gt_transform=True,
        x_is_single_channel=True,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='RGBXIoUMetric',
    ignore_index=255,
    iou_metrics=['mIoU'],
    num_classes=num_classes)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, weight_decay=0.01),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        begin=0,
        end=warmup_iters,
        by_epoch=False,
    ),
    dict(
        type='PolyLR',
        eta_min=1e-7,
        power=0.9,
        begin=warmup_iters,
        end=max_iters,
        by_epoch=False,
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=max(1, iters_per_epoch // 10), log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=checkpoint_interval, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

seed = 12345
work_dir = 'work_dirs/dformer/dformerv2_b_8xb16-500e_nyudepthv2-480x640'

# Legacy-aligned evaluation settings used by tools/test.py advanced mode.
background = 255
eval_stride_rate = 2 / 3
eval_crop_size = (image_height, image_width)
if dataset_name == 'SUNRGBD':
    eval_scale_array = [0.5, 0.75, 1.0, 1.25, 1.5]
else:
    eval_scale_array = [1.0]
eval_flip = True
