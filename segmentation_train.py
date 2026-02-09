import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torch.utils.data import DataLoader
import cv2
import os
import json
from imantics import Mask

from pycocotools.cocoeval import COCOeval
from torchvision.transforms import functional as F
from pycocotools import mask as maskUtils
import csv
import dataset as ds
from multiprocessing import Process, freeze_support, set_start_method
import math
from utils import extract_bboxes
#Import Mask R-CNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


INTEL_SIZE = (1280, 720)

def train():
    #class_map = {}
    #with open("C:/Users/alecr/OneDrive/Documents/GitHub/TACO/detector/taco_config/map_RRR.csv") as csvfile:
        #reader = csv.reader(csvfile)
        #class_map = {row[0]:row[1] for row in reader}

    #TACO_DIR = "C:/Users/alecr/OneDrive/Documents/GitHub/TACO/data/"
    #AUG_TACO_DIR = "C:/Users/alecr/OneDrive/Documents/GitHub/TACO/data/annotations_augmented"
    #round = None
    #subset = "train"
    #dataset = ds.Taco()
    #taco = dataset.load_taco(TACO_DIR, round, subset, class_map = class_map, return_taco=True)


    #model = maskrcnn_resnet50_fpn(pretrained=True)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    def target_transform(targets):
        ret = []
        for target in targets:
            boxes = target['bbox']
            boxes = torch.tensor(boxes, dtype = torch.float32)
            boxes = torch.reshape(boxes, (-1, 4))
            labels = torch.tensor([target['category_id']], dtype = torch.int64)
            segm = target['rle']
            #print("Segm: " + str(segm))
            rles = maskUtils.frPyObjects(segm, INTEL_SIZE[1], INTEL_SIZE[0])
            rle = maskUtils.merge(rles)
            mask = maskUtils.decode(rle)
            masks = torch.tensor([mask], dtype = torch.uint8)
            image_id = torch.tensor([target['image_id']])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes), ), dtype = torch.int64)
            ret.append({'boxes': boxes, 'labels': labels, 'masks': masks})
        return ret
    
    
    dataset = torchvision.datasets.CocoDetection('C:/Users/alecr/OneDrive/Documents/GitHub/TACO/data', 'C:/Users/alecr/OneDrive/Documents/GitHub/TACO/data/resize_annotations.json', transform = transform, target_transform= target_transform)

    
    image, target = dataset[12]
    transform = transforms.ToPILImage()
    plt.imshow(transform(image))
    plt.show()
    print(target)

    cv2.waitKey(0)



    print("DATA WAS COCO'd")
    #Create data loader.
    #data_loader = DataLoader(dataset, batch_size = 10, shuffle = True, collate_fn = lambda batch: tuple(zip(*batch)))
    print("DATA LOADER was finished")
    '''
    #Initialize a Mask R-CNN model with pretrained weights. 
    model = maskrcnn_resnet50_fpn_v2(pretrained = True)

    #Load parameters and optimizer. 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)

    num_epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    imageTransform = transforms.ToPILImage()
    #plt.imshow(imageTransform(image))
    #plt.imshow(imageTransform(target['masks']), alpha = 0.5)
    #plt.show()
    #cv2.waitKey()

    for epoch in range(num_epochs):
        for images, targets in data_loader:
            #print("Images in Batch: " + str(images))
            #print("Targets in Batch: " + str(targets))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            print(losses.item())
                
        lr_scheduler.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}')
    
    torch.save(model.state_dict(), 'model_weights.pth')

    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
    



def resize():
    imgs = set()
    class_map = {}
    with open("C:/Users/alecr/OneDrive/Documents/GitHub/TACO/detector/taco_config/map_RRR.csv") as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]:row[1] for row in reader}

    TACO_DIR = "C:/Users/alecr/OneDrive/Documents/GitHub/TACO/data/"
    RESIZE_TACO_DIR = "C:/Users/alecr/OneDrive/Documents/GitHub/TACO/data/annotations_resize"
    round = None
    subset = "train"
    dataset = ds.Taco()
    taco = dataset.load_taco(TACO_DIR, round, subset, class_map = class_map, return_taco=True)
    dataset.prepare()
    print("CLASS NAMES: " + str(dataset.class_names))
    imgs = set()
    imageCount = 0
    annotationCount = 0
    
    annotations_file_path = os.path.join(TACO_DIR, 'annotations.json')
    assert os.path.isfile(annotations_file_path)

    with open(annotations_file_path, 'r') as f:
        annotations_json = json.loads(f.read())
    #print(annotations_json.keys())
    new_annotations = annotations_json.copy()
    new_annotations['images'].clear()
    new_annotations['annotations'].clear()
    new_annotations['scene_annotations'].clear()
    #new_annotations['']
    #print("New annotations: {}".format(new_annotations))
    
    #ITERATE THROUGH
    for ann in taco.dataset['annotations']:
        #If annotation is for background, IGNORE. 
        
        categoryID = ann['category_id']
        #If image has already been transformed, IGNORE. 
        imageID = ann['image_id']
        if(imageID in imgs):
            continue
        else:
            imgs.add(imageID)
            imageCount = imageCount + 1
            print(imageCount)
        
        image_anns = taco.loadAnns(taco.getAnnIds(imageID, taco.getCatIds(), iscrowd=None))
        image = dataset.load_image(imageID)
        orig_height, orig_width, _ = image.shape
        file_name = "resized/000" + str(imageCount) + ".jpg"
        path = 'C:/Users/alecr/OneDrive/Documents/GitHub/TACO/data/'+file_name
        
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
        

        #print("Image saved {}, {}".format(resized_image.shape[0], resized_image.shape[1]))
        cv2.imwrite(path, cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        rowsResize, colsResize, _ = resized_image.shape        
        new_image = {'id': imageCount, 'width': colsResize, 'height': rowsResize, 'file_name':file_name, 'license':None, 'flicker_url': None, 'coco_url': None, 'data_captured': None, 'flicker_640_url': None}
        new_annotations['images'].append(new_image)

        #Then add all transformed annotations. 
        for image_ann in image_anns:
            annotationCount = annotationCount + 1
            segmentation = image_ann['segmentation']
            #print("Seg: " + str(segmentation))
            categoryID = image_ann['category_id']
            #print("OLD SEGMENTATION: " + str(segmentation))
            rles = maskUtils.frPyObjects(segmentation, orig_height, orig_width)
            rle = maskUtils.merge(rles)
            a_mask = maskUtils.decode(rle)

            #Rotate and resize. 
            if(rotate_flag):
                a_mask = cv2.rotate(a_mask, cv2.ROTATE_90_CLOCKWISE)
            
            resized_mask = cv2.resize(a_mask, None, fx = scale_factor, fy = scale_factor, interpolation = cv2.INTER_LINEAR)
            
            if(padding_flag == 0):
                resized_mask = cv2.copyMakeBorder(resized_mask, math.floor(top_padding/2), math.ceil(top_padding/2), math.floor(left_padding/2), math.ceil(left_padding/2), cv2.BORDER_CONSTANT, None, 0)

            #plt.imshow(resized_image)
            #plt.imshow(resized_mask, alpha = 0.5)
            #plt.show()
            #cv2.waitKey()

            plt.figure(1)
            #plt.imshow(resized_image)
            #plt.imshow(resized_mask, alpha = 0.5)
            #plt.show()
            #cv2.waitKey()
            
            '''
            firstcontours, heirarchy = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = []
            #print("XVals: " + str(x_values))
            #print("YVals: " + str(y_values))
            #contours = contours.reshape(1, -1)
            #new_segmentation.append(contours[0].tolist())
            print("Contours: " + str(firstcontours))
            for obj in firstcontours:
                for point in obj:
                    polygon.append([point[0][0], point[0][1]])
            polygon = np.asarray(polygon)
            polygon = polygon.reshape(-1, 2)
            x_values = polygon[:, 0]
            y_values = polygon[:, 1]
            #print("X_values: " + str(x_values))
            x_min = min(x_values)
            x_max = max(x_values)
            y_min = min(y_values)
            y_max = max(y_values)
            '''

            polygons = Mask(resized_mask).polygons()
            new_bbox = Mask(resized_mask).bbox()

            
            new_bbox = list([new_bbox._xmin, new_bbox._ymin, new_bbox._xmax, new_bbox._ymax])
            #print("OLD: " + str(polygons.segmentation))
            #Prevent segmentations pretending to be bounding boxes
            segmentation = []
            for segm in polygons.segmentation:
                if len(segm) > 4:
                    segmentation.append(segm)
            #print("NEW: " + str(segmentation))
            try:
                rles = maskUtils.frPyObjects(segmentation, INTEL_SIZE[1], INTEL_SIZE[0])
                new_ann = {'id': annotationCount, 'image_id': imageCount, 'category_id':categoryID, 'rle': segmentation, 'bbox': new_bbox}
                new_annotations['annotations'].append(new_ann)
            except:
                print(segmentation)
            #rle = maskUtils.merge(rles)
            #mask = maskUtils.decode(rle)
            #plt.imshow(resized_image)
            #plt.imshow(mask, alpha = 0.5)
            #plt.show()
            #cv2.waitKey()
            

            #print(new_bbox)
            
            #json_str = json.dumps(new_annotations, indent=7)


    json_str = json.dumps(new_annotations, indent=7)
    #print("Json STR: " + str(json_str))
    with open("C:/Users/alecr/OneDrive/Documents/GitHub/TACO/data/resized/annotations.json", "w") as f:
        f.write(json_str)

    

    '''

    ann = taco.dataset['annotations'][0]
    imgID = ann['image_id']
    image = dataset.load_image(imgID)
    print("Shape: {}".format(image.shape))
    fx = INTEL_SIZE[0] / image.shape[0]
    fy = INTEL_SIZE[1] / image.shape[1]
    print("Fx: {}, Fy:{}".format(fx, fy))
    image_ann = taco.loadAnns(taco.getAnnIds(imgID, taco.getCatIds(), iscrowd = None))
    ann = image_ann[0]
    segmentation = ann['segmentation']

    new_seg = np.asarray(segmentation).reshape(int(len(segmentation[0])/2), -1)
    scale_mat = np.asarray([[fx, 0], [0, fy]])
    scale_seg = np.matmul(new_seg, scale_mat).reshape(-1, len(segmentation[0]))

    


    categoryID = ann['category_id']
    print("Segmentation:{}".format(segmentation))

    #plt.figure(1)
    point1 = (400, 400)
    cv2.circle(image, center = point1, radius = 5, color = (0, 0, 255), thickness = -1)
    cv2.imshow('Image1', image)


    #plt.figure(2)
    resized_image = cv2.resize(image,None, fx = fx, fy = fy, interpolation = cv2.INTER_LINEAR)
    point2 = (int(400*fx), int(400*fy))
    cv2.circle(resized_image, center = point2, radius = 4, color = (0, 255, 0), thickness = -1)
    cv2.imshow('Image2', resized_image)
    cv2.imwrite(path, cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

    cv2.waitKey()
    #plt.show()
    '''



if __name__ == '__main__':
    freeze_support()
    set_start_method('spawn')
    p = Process(target =train)
    p.start()