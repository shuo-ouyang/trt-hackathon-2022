import numpy as np
import os
import torch
from dataset import build_dataset, build_transform
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img-val', type=str, default='~/imagenet/val', help='imagenet validation data path')
args = parser.parse_args()

transform = build_transform()
dataset, _ = build_dataset(args.img_val, transform)
sampler = torch.utils.data.SequentialSampler(dataset)
data_loader_val = torch.utils.data.DataLoader(
    dataset,
    sampler=sampler,
    batch_size=int(32),
    num_workers=1,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
    shuffle=False,
)
cnt = 0
for idx, (images, labels) in enumerate(data_loader_val):
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    np.savez(f"./calibration/calib-{idx}", images=images, labels=labels)
    cnt += images.shape[0]
    if cnt >= 5120:
        break
