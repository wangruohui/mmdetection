_base_ = [
    "fast_rcnn_r18.py",
    "voc07_proposal.py",
    "../_base_/default_runtime.py",
]
# optimizer
optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy="step", step=[3])
# runtime settings
total_epochs = 4  # actual epoch = 4 * 3 = 12
