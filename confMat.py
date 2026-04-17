import pandas as pd
import numpy as np
import os

import sys
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools import mask as maskUtils
import numpy as np
import argparse
from multiprocessing import Process, freeze_support, set_start_method
import matplotlib.pyplot as plt
from engine import train_one_epoch, evaluate
from torchvision import tv_tensors
import json
import os
from torchvision.transforms import v2 as T
from torch.utils.tensorboard import SummaryWriter
import cv2
import confUtils
INTEL_SIZE = (1280, 720)
from test import TACODataset, get_transform, get_model_instance_segmentation, custom_loader
gt_tot = np.array([])
pred_tot = np.array([])
path = 'data/resized'
model = get_model_instance_segmentation(7)
model.load_state_dict(torch.load('models/model_weights10_10.pth', weights_only=True,  map_location=torch.device('cpu')))
model.eval()
dataset = TACODataset(path, get_transform(True))
data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True, collate_fn = custom_loader)
count = 0
for img, target in data_loader:
    img = img[0]
    target = target[0]
    gt_bbox = target['boxes']
    gt_masks = target['masks']
    gt_classes = target['labels']

    results = model([img])[0]

    boxes = np.int64(results['boxes'].detach().numpy())
    masks = results['masks'].detach().permute(0, 2, 3, 1)
    labels = np.int8(results['labels'].detach())
    scores = results['scores'].detach()

    gt, pred = confUtils.gt_pred_lists(gt_classes, gt_bbox, labels, boxes)
    gt_tot = np.append(gt_tot, gt)
    pred_tot = np.append(pred_tot, pred)
    count = count + 1
    print("Count: ", count)
    if(count > 100):
        break

gt_tot = gt_tot.astype(int)
pred_tot = pred_tot.astype(int)
print(gt_tot)
print(pred_tot)
tp, fp, fn = confUtils.plot_confusion_matrix_from_data(gt_tot, pred_tot, fz=18, figsize=(20, 20), lw = 0.5)
