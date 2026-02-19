import sys
import torchvision
import torchvision.transforms as transforms
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools import mask as maskUtils
import numpy as np
import argparse
from multiprocessing import Process, freeze_support, set_start_method
import matplotlib.pyplot as plt
import cv2
import math
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch.nn.functional as F
from torchvision.tv_tensors import BoundingBoxes, Mask
from distinctipy import distinctipy
import visualize
import test


INTEL_SIZE = (1280, 720)
mean = torch.tensor([0.3825, 0.3623, 0.3205]) 
std = torch.tensor([0.3010, 0.2854, 0.2710])

transform = transforms.Compose([
        transforms.ToTensor()])


image = cv2.imread('data/test/007.jpg')


orig_height, orig_width, _ = image.shape

#TODO:
if orig_height > orig_width:
    rotate_flag = True
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  
    scale_factor = INTEL_SIZE[1]/image.shape[0]
else:
    rotate_flag = False    
    #print("This occurred")
    scale_factor = INTEL_SIZE[1]/image.shape[0]



resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
#print("New shape: " + str(resized_image.shape))
if(resized_image.shape[1] > INTEL_SIZE[0]):
    scale_factor = scale_factor* INTEL_SIZE[0]/resized_image.shape[1]
    resized_image = cv2.resize(image, None, fx = scale_factor, fy = scale_factor, interpolation = cv2.INTER_LINEAR)

#Pad on the sides.
if(INTEL_SIZE[0] > resized_image.shape[1] or INTEL_SIZE[1] > resized_image.shape[0]):
    padding_flag = 0
    left_padding = INTEL_SIZE[0] - resized_image.shape[1]
    top_padding = INTEL_SIZE[1] - resized_image.shape[0]
    resized_image = cv2.copyMakeBorder(resized_image, math.floor(top_padding/2), math.ceil(top_padding/2), math.floor(left_padding/2), math.ceil(left_padding/2), cv2.BORDER_CONSTANT, None, 0)
else:
    padding_flag = -1

model = test.get_model_instance_segmentation(7)
#model = maskrcnn_resnet50_fpn_v2(weights = 'DEFAULT')
#in_features_box = model.roi_heads.box_predictor.cls_score.in_features
#in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

#dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

#model.roi_heads.box_predictor = FastRCNNPredictor(in_channels = in_features_box, num_classes = 7)
#model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels = 256, dim_reduced=256, num_classes = 7)
model.load_state_dict(torch.load('models/model_weights10_10.pth', weights_only=True,  map_location=torch.device('cpu')))
model.eval()

resized_image = np.uint8(resized_image)
class_names_17 = ['BG', 'Aluminium foil', 'Can', 'Carton', 'Cup', 'Glass bottle', 'Metal bottle cap', 'Other', 'Paper', 'Plastic bottle', 'Plastic bottle cap', 'Plastic container', 'Plastic film', 'Plastic lid', 'Pop tab', 'Straw', 'Styrofoam piece', 'Wrapper']
class_names = ['BG', 'Glass', 'Metal', 'Other', 'Paper', 'Plastic', 'Trash']


resized_image_tensor = transform(resized_image)

results = model([resized_image_tensor])[0]

boxes = np.int64(results['boxes'].detach().numpy())
masks = results['masks'].detach().permute(0, 2, 3, 1)
labels = np.int8(results['labels'].detach())
scores = results['scores'].detach()

colors = distinctipy.get_colors(len(class_names))

visualize.visualize(resized_image, masks, boxes, labels, class_names,scores, colors, .4, 100)
