RGB_LAYERS = 50
POSE_LAYERS = 18
FRAME_IDS = [0, -10, 10]
IMGS_PER_GPU = 12
HEIGHT = 192  # 288
WIDTH = 256  # 384

data = dict(
    name='mipi',
    split='mipi',
    height=HEIGHT,
    width=WIDTH,
    frame_ids=FRAME_IDS,
    in_path='/home/arthur/workspace/Datasets/MIPI2022/train',
    gt_depth_path=None,
    png=False,
    stereo_scale=False,
)

model = dict(
    name='emdc_fm_nml',
    rgb_num_layers=RGB_LAYERS,
    pose_num_layers=POSE_LAYERS,
    frame_ids=FRAME_IDS,
    imgs_per_gpu=IMGS_PER_GPU,
    height=HEIGHT,
    width=WIDTH,
    scales=[0, 1, 2, 3],
    min_depth=1e-3,
    max_depth=10,
    depth_pretrained_path='/home/arthur/workspace/Projects/FeatDepth/weights/mobilenetv2.pth.tar',
    rgb_pretrained_path='/home/arthur/workspace/Projects/FeatDepth/weights/resnet/resnet{}.pth'.format(RGB_LAYERS),
    pose_pretrained_path='/home/arthur/workspace/Projects/FeatDepth/weights/resnet/resnet{}.pth'.format(POSE_LAYERS),
    normal_pretrained_path='/home/arthur/workspace/Projects/depthcompletiontoolbox/weights/normal_nyuv2.pt',
    extractor_pretrained_path='./../weights/autoencoder.pth',
    automask=False if 's' in FRAME_IDS else True,
    disp_norm=False if 's' in FRAME_IDS else True,
    dis=1e-3,
    cvt=1e-3,
    perception_weight=1e-3,
    smoothness_weight=1e-3,
)

# resume_from = '/home/arthur/workspace/Projects/FeatDepth-master/fmdepth/epoch_9.pth'
resume_from = '/home/arthur/workspace/Projects/depthcompletiontoolbox/EXP_EMDC_FM_NML/epoch_36.pth'
# resume_from = None
finetune = None
total_epochs = 150
imgs_per_gpu = IMGS_PER_GPU
learning_rate = 1e-3
workers_per_gpu = 4
validate = False
by_epoch = True
optimizer = dict(type='AdamW', lr=learning_rate, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnealing',
    warmup='linear',
    warmup_iters=10 if by_epoch else 1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5,
    by_epoch=by_epoch,
    warmup_by_epoch=by_epoch)

checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
workflow = [('train', 1)]
