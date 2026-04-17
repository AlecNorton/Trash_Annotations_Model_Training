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
from math import atan2, cos, sin, sqrt, pi


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


INTEL_SIZE = (1280, 720)
mean = torch.tensor([0.3825, 0.3623, 0.3205]) 
std = torch.tensor([0.3010, 0.2854, 0.2710])

transform = transforms.Compose([
        transforms.ToTensor()])
model = test.get_model_instance_segmentation(7)
#model = maskrcnn_resnet50_fpn_v2(weights = 'DEFAULT')
#in_features_box = model.roi_heads.box_predictor.cls_score.in_features
#in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

#dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

#model.roi_heads.box_predictor = FastRCNNPredictor(in_channels = in_features_box, num_classes = 7)
#model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels = 256, dim_reduced=256, num_classes = 7)
model.load_state_dict(torch.load('models/model_weights5_10_finetune.pth', weights_only=True,  map_location=torch.device('cpu')))
model.eval()
class_names_17 = ['BG', 'Aluminium foil', 'Can', 'Carton', 'Cup', 'Glass bottle', 'Metal bottle cap', 'Other', 'Paper', 'Plastic bottle', 'Plastic bottle cap', 'Plastic container', 'Plastic film', 'Plastic lid', 'Pop tab', 'Straw', 'Styrofoam piece', 'Wrapper']
class_names = ['BG', 'Glass', 'Metal', 'Other', 'Paper', 'Plastic', 'Trash']

for i in range(50, 101):
    image = cv2.imread('data/test/images/0000'+str(i)+'.jpg')
    print("I: ", i)



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

    

    resized_image = np.uint8(resized_image)
    


    resized_image_tensor = transform(resized_image)

    results = model([resized_image_tensor])[0]
    boxes = np.int64(results['boxes'].detach().numpy())
    masks = results['masks'].detach().permute(0, 2, 3, 1)
    labels = np.int8(results['labels'].detach())
    scores = results['scores'].detach()

    colors = distinctipy.get_colors(len(class_names))

    img, combinedMask, color_masks, new_labels, new_boxes, new_masks = visualize.visualize(resized_image, masks, boxes, labels, class_names,scores, colors, 0.10, 100)
    #cv2.imshow('combinedMask', combinedMask)
    #cv2.imshow('img', resized_image)
    #cv2.waitKey(0)
    
    for mask in new_masks:
        mask = mask*255
        #cv2.imshow('masks', mask)
        #cv2.waitKey(0)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(mask, contours, -1, (0, 0, 255), 2)
        data_points = contours[0]
        if(np.shape(data_points)[0] < 2):
            continue
        #Do PCA analysis. 
        sz = len(data_points)
        data_pts = np.empty((sz, 2), dtype = np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = data_points[i, 0, 0]
            data_pts[i, 1] = data_points[i, 0, 1]
        #print("Data Points: ", data_points)
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
        #print("Eigenvectors: ", eigenvectors)
        cntr = (int(mean[0, 0]), int(mean[0, 1]))

        cv2.circle(img, cntr, 3, (255, 0, 255), 2)

        p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
        drawAxis(img, cntr, p1, (0, 255, 0), 1)
        drawAxis(img, cntr, p2, (255, 255, 0), 5)
        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        print(angle)
    print("Done")
    cv2.imshow('img', img)
    cv2.imshow('act_img', resized_image)
    cv2.waitKey(0)
    