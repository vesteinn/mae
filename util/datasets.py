# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import random
from collections import defaultdict


def filter_dataset(dataset, keep_ratio=1):
    # Remove percentage of jpgs!
    new_samples = []
    new_targets = []

    jpgs_samples = defaultdict(list)
   
    for idx, sample in enumerate(dataset.samples):
        if sample[0].endswith("newclas.jpg"):
            new_samples.append(sample)
            continue
        jpgs_samples[sample[1]].append(sample[0])
    new_targets = [sample[1] for sample in new_samples]

    for c_name, samples in jpgs_samples.items():
        k = int(len(samples) * keep_ratio)
        random.seed(0)
        formated_samples = [(s, c_name) for s in samples]
        randsamp = random.sample(formated_samples, k)
        new_samples += randsamp
        new_targets += [s[1] for s in randsamp]
    
    print(f"{len(new_samples)} -- {len(new_targets)}")
    assert len(new_samples) == len(new_targets)
    dataset.samples = new_samples
    dataset.targets = new_targets
    dataset.imgs = new_samples
    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    if args.keep_orig_ratio:
        dataset = filter_dataset(dataset, keep_ratio=args.keep_orig_ratio)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
