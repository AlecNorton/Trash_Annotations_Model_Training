print("Loading libraries...")
import torchvision.transforms as transforms
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn
from pycocotools import mask as maskUtils
import numpy as np
import cv2
import math
from distinctipy import distinctipy
import os
import test
import visualize
from imantics import Mask
import json
from rich.progress import Progress


class_names = ['BG', 'Glass', 'Metal', 'Other', 'Paper', 'Plastic', 'Trash']
score_threshold = .25
size_threshold = 30
colors = distinctipy.get_colors(len(class_names))

model = test.get_model_instance_segmentation(7)
model.load_state_dict(torch.load('models/model_weights10_10.pth', weights_only=True,  map_location=torch.device('cpu')))
model.eval()

class_names = ['BG', 'Glass', 'Metal', 'Other', 'Paper', 'Plastic', 'Trash']
INTEL_SIZE = (1280, 720)
mean = torch.tensor([0.3825, 0.3623, 0.3205]) 
std = torch.tensor([0.3010, 0.2854, 0.2710])

transform = transforms.Compose([
        transforms.ToTensor()])

ann_count = 0
img_count = 0
#Get images from data/test
resized_img_tensors = []
resized_imgs = []
new_annotations = {}
breakFlag = False
with Progress() as P:
    ls = os.listdir("C:/Users/alecr/OneDrive/Documents/GitHub/Trash_Annotations_Model_Training/data/test/images")
    numImages = len(ls)-1

    print(numImages)
    t = P.add_task("Preprocesing images...", total = numImages)
    for img_path in ls:

        if breakFlag == True:
            break
        if(img_path.endswith(".jpg")):
            img_count= img_count+1
            new_annotations[img_count-1] = []

            img = cv2.imread("C:/Users/alecr/OneDrive/Documents/GitHub/Trash_Annotations_Model_Training/data/test/images/" + img_path)
            orig_height, orig_width, _ = img.shape
            
            #Rotate image
            if orig_height > orig_width:
                rotate_flag = True
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  
                scale_factor = INTEL_SIZE[1]/img.shape[0]
            else:
                rotate_flag = False    
                #print("This occurred")
                scale_factor = INTEL_SIZE[1]/img.shape[0]

            #Resize image.
            resized_image = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            if(resized_image.shape[1] > INTEL_SIZE[0]):
                scale_factor = scale_factor* INTEL_SIZE[0]/resized_image.shape[1]
                resized_image = cv2.resize(img, None, fx = scale_factor, fy = scale_factor, interpolation = cv2.INTER_LINEAR)

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

            #Add them to list
            resized_img_tensors.append(resized_image_tensor)
            resized_imgs.append(resized_image)
            P.update(t, advance = 1)

print("END OF PROCESSING: ", img_count)
#json_str = json.dumps(new_annotations, indent=7)
#print("Json STR: " + str(json_str))
print("is this saving at all or")
for img in resized_imgs:
    print("HELLO??")
    cv2.imshow('orig_img', img)
    cv2.waitKey()
#with open("C:/Users/alecr/OneDrive/Documents/GitHub/Trash_Annotations_Model_Training/data/test/annotations.json", "w") as f:
#    f.write(json_str)


print("Putting first 10 imgs into model... just wait...")
ann_count = 0
img_count = 0
i = 0

#with open("C:/Users/alecr/OneDrive/Documents/GitHub/Trash_Annotations_Model_Training/data/test/annotations.json", "r") as f:
#    new_annotations = json.load(f)

print("new_ann: ", new_annotations)
results = model(resized_img_tensors[i*10:(i*10)+10])
for result in results:
    resized_image = resized_imgs[img_count]
    img_count = img_count + 1

    boxes = np.int64(result['boxes'].detach().numpy())
    masks = result['masks'].detach().permute(0, 2, 3, 1)
    labels = np.int8(result['labels'].detach())
    scores = result['scores'].detach()

    idx = np.where(scores >= score_threshold)
    masks = masks[idx]
    boxes = boxes[idx]
    scores = scores[idx]
    labels = labels[idx]
    
    combinedMask, color_masks, new_labels, new_boxes, new_masks = visualize.convert_masks(masks, labels, colors, boxes, size_threshold)
    for mask in color_masks:
        if breakFlag == True:
            break
        mask = np.uint8(mask*255)
        
        res = cv2.addWeighted(resized_image, .25, np.uint8(mask)*255, .75, 0)
        cv2.imshow('origImage', resized_image)
        cv2.imshow('res', res)
        print("1 - Glass, 2 - Metal, 3 - Other, 4 - Paper, 5 - Plastic, 6 - Trash, c - ignore annotation")
        key = cv2.waitKey(0)
        print("Key: ", key)
        
        match key:
            case x if x >= ord('0') and x <= ord('9'):
                print("Labeling annotation.")
                ann_count = ann_count + 1
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                
                polygons = Mask(mask).polygons()
                new_bbox = Mask(mask).bbox()
                new_bbox = list([new_bbox._xmin, new_bbox._ymin, new_bbox._xmax, new_bbox._ymax])
                segmentation = []
                for segm in polygons.segmentation:
                    if len(segm) > 4:
                        segmentation.append(segm)
                try:
                    rles = maskUtils.frPyObjects(segmentation, INTEL_SIZE[1], INTEL_SIZE[0])
                    new_ann = {'id': ann_count, 'image_id': img_count-1, 'category_id':abs(ord('0') - x), 'rle': segmentation, 'bbox': new_bbox}
                    new_annotations[str(img_count-1)].append(new_ann)
                except:
                    print(segmentation)
            case x if x == ord('c'):
                #Continue, do not read mask into annotations.
                #print("Ignoring annotation")
                continue
            case x if x == ord('q'):
                #print(new_annotations)
                print("Getting final annotation block!")
                breakFlag = True


print("Image count: ", img_count)
print("Annotation Count: ", ann_count)
json_str = json.dumps(new_annotations, indent=7)

with open("C:/Users/alecr/OneDrive/Documents/GitHub/Trash_Annotations_Model_Training/data/test/annotations.json", "w") as f:
    f.write(json_str)
                
                

                    
                    
                    

            




