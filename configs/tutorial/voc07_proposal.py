# dataset settings
dataset_type = "VOCDataset"
data_root = "data/VOCdevkit/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadProposals", num_max_proposals=2000),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1000, 600), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "proposals", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadProposals", num_max_proposals=None),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="ToTensor", keys=["proposals"]),
            dict(
                type="ToDataContainer",
                fields=[dict(key="proposals", stack=False)],
            ),
            dict(type="Collect", keys=["img", "proposals"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "VOC2007/ImageSets/Main/train.txt",
            img_prefix=data_root + "VOC2007/",
            proposal_file=data_root
            + "proposals/rpn_r50_fpn_1x_voc2007_train.pkl",
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "VOC2007/ImageSets/Main/val.txt",
        img_prefix=data_root + "VOC2007/",
        proposal_file=data_root + "proposals/rpn_r50_fpn_1x_voc2007_val.pkl",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "VOC2007/ImageSets/Main/test.txt",
        img_prefix=data_root + "VOC2007/",
        proposal_file=data_root + "proposals/rpn_r50_fpn_1x_voc2007_test.pkl",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="mAP")
