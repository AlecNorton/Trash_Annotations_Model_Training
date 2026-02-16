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
INTEL_SIZE = (1280, 720)
mean = torch.tensor([0.3825, 0.3623, 0.3205])
std = torch.tensor([0.3010, 0.2854, 0.2710])


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    #transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
 
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes)
    return model

class TACODataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
 
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = json.load(open(os.path.join(root, "annotations.json")))

    def __getitem__(self, idx):
        
        img_id = self.annotations[str(idx)][0]['image_id']
        img_path = os.path.join(self.root, "images", "000" + str(img_id+1) + ".jpg")
        img = read_image(img_path)
        boxes = []
        labels = []
        masks = []
        numObjs = 0
        for obj in self.annotations[str(idx)]:
            numObjs = numObjs + 1
            boxes.append(obj['bbox'])
            segm = obj['rle']
            rles = maskUtils.frPyObjects(segm, INTEL_SIZE[1], INTEL_SIZE[0])
            rle = maskUtils.merge(rles)
            mask = maskUtils.decode(rle)
            masks.append(mask)
            labels.append(obj['category_id'])
            image_id = obj['image_id']
        boxes = torch.tensor(np.array(boxes))
        #print(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((numObjs, ), dtype = torch.int64)
        img = tv_tensors.Image(img)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=INTEL_SIZE)
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = torch.tensor(labels)
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    def __len__(self):
        return len(self.imgs)

'''
def target_transform(targets):
    ret = {'boxes': [], 'labels': [], 'masks': [], 'image_id': [], 'area': [], 'iscrowd':[]}
    for target in targets:
        boxes = target['bbox']
        #boxes = torch.tensor(boxes, dtype = torch.float32)
        labels = target['category_id']
        segm = target['rle']
        #print("Segm: " + str(segm))
        rles = maskUtils.frPyObjects(segm, INTEL_SIZE[1], INTEL_SIZE[0])
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)
        area = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
        #masks = torch.tensor(mask, dtype = torch.uint8)
        boxes = tv_tensors.BoundingBoxes(boxes, format = "XYXY", canvas_size = (720, 1280))
        mask = tv_tensors.Mask(mask)
        ret['boxes'].append(boxes)
        ret['labels'].append(labels)
        ret['masks'].append(mask)
        ret['image_id'].append(target['image_id'])
        ret['area'].append(area)
        ret['iscrowd'].append(0)
    #print(ret['boxes'])
    ret['labels'] = torch.tensor(ret['labels'], dtype = torch.int64)
    #ret['masks'] = torch.tensor(ret['masks'])
    ret['image_id'] = torch.tensor(ret['image_id'], dtype = torch.int64)
    ret['area'] = torch.tensor(ret['area'], dtype = torch.float32)
    ret['iscrowd'] = torch.tensor(ret['iscrowd'], dtype = torch.uint8)
    return ret
'''
def custom_loader(batch):
    return tuple(zip(*batch))

def class_distribution():
    dataset = TACODataset('data/resized', get_transform(True))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 4, collate_fn = custom_loader)

    count = 0
    class_labels = torch.zeros(18)
    t = transforms.ToPILImage()
    print("Len, ", len(dataset))
    count = 0
    for imgs, targets in data_loader:
        print(count)
        count +=1
        #plt.imshow(t(imgs))
        #plt.imshow(np.uint8(targets['masks'][0]))
        #plt.show()
        #cv2.waitKey()
        #print("Labels: ", targets['labels'])
        #print(targets)
        add = np.bincount(targets[0]['labels'])
        #print(len(add))
        add = np.pad(add, (0, 18 - len(add)))
        class_labels = class_labels + add
        #print("Class_Labels: ", class_labels)
    print("Class labels,", class_labels)
    

def train(num_epochs, batch_size, data):

    print("\nLoading dataset.\n")

    if(data == 'RRR'):
        path = 'data/resized'
        string = 'RRR'
    elif(data == '17'):
        path = 'data/resized_17'
        string = '17'
    else:
        raise ValueError("Wrong data submission.")

    #dataset = torchvision.datasets.CocoDetection('data', 'data/resized/annotations.json', transform = transform, target_transform = target_transform)
    dataset = TACODataset(path, get_transform(True))
    dataset_test = TACODataset(path, get_transform(False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-150])
    test_dataset = torch.utils.data.Subset(dataset_test, indices[-10:])
    print("Complete dataset load.")
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = custom_loader)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, collate_fn = custom_loader)
    print("Train data loader initialized.")

    model = maskrcnn_resnet50_fpn_v2(weights = 'DEFAULT')

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels = in_features_box, num_classes = 7)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels = in_features_mask, dim_reduced = dim_reduced, num_classes = 7)

    print("Model loaded.")

    #Load parameters and optimizer. 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)

    print("Optimizer loaded.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    #plt.imshow(imageTransform(image))
    #plt.imshow(imageTransform(target['masks']), alpha = 0.5)
    #plt.show()
    #cv2.waitKey()

    #print("DOES THIS WORK?",test_dataset.__getitem__(0))
    print("EPOCHS: " + str(num_epochs))
    print("BATCH_SIZE: " + str(batch_size))
    
    writer = SummaryWriter(log_dir = 'runs/'+str(num_epochs)+'_'+str(batch_size)+'_'+string)
    
    for epoch in range(num_epochs):
        print("EPOCH: ", epoch)
        
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq = 10)
        
        writer.add_scalar("Train/LR", metric_logger.__getattr__('lr').median, epoch)
        writer.add_scalar("Train/Loss", metric_logger.__getattr__('loss').median, epoch)
        writer.add_scalar("Train/Loss_classifier", metric_logger.__getattr__('loss_classifier').median, epoch)
        writer.add_scalar("Train/Loss_box_reg", metric_logger.__getattr__('loss_box_reg').median, epoch)
        writer.add_scalar("Train/Loss_mask", metric_logger.__getattr__('loss_mask').median, epoch)
        writer.add_scalar("Train/Loss_objectness", metric_logger.__getattr__('loss_objectness').median, epoch)
        writer.add_scalar("Train/Loss_rpn_box_reg", metric_logger.__getattr__('loss_rpn_box_reg').median, epoch)
        
        #print("METRIC LOG", metric_logger.__str__())
        #print("\nMETIRC LOG", metric_logger.meters)
        #print("METRIC LOG", metric_logger.loss)
        
        lr_scheduler.step()
        evaluator = evaluate(model, data_loader_test, device=device)
        for iou_type, coco_eval in evaluator.coco_eval.items():
            writer.add_scalar("AP/" + str(iou_type)+"/.50-.95_all", coco_eval.stats[0], epoch)
            writer.add_scalar("AP/" + str(iou_type)+"/.50_all", coco_eval.stats[1], epoch)
            writer.add_scalar("AP/" + str(iou_type)+"/.75_all", coco_eval.stats[2], epoch)
            writer.add_scalar("AP/" + str(iou_type)+"/.50-95_small", coco_eval.stats[3], epoch)
            writer.add_scalar("AP/" + str(iou_type)+"/.50-95_medium", coco_eval.stats[4], epoch)
            writer.add_scalar("AP/" + str(iou_type)+"/.50-95_large", coco_eval.stats[5], epoch)
            writer.add_scalar("AR/" + str(iou_type)+"/.50-.95_all", coco_eval.stats[6], epoch)
            writer.add_scalar("AR/" + str(iou_type)+"/.50_all", coco_eval.stats[7], epoch)
            writer.add_scalar("AR/" + str(iou_type)+"/.75_all", coco_eval.stats[8], epoch)
            writer.add_scalar("AR/" + str(iou_type)+"/.50-95_small", coco_eval.stats[9], epoch)
            writer.add_scalar("AR/" + str(iou_type)+"/.50-95_medium", coco_eval.stats[10], epoch)
            writer.add_scalar("AR/" + str(iou_type)+"/.50-95_large", coco_eval.stats[11], epoch)
        '''
        data_iter = iter(data_loader)
        for images, targets in data_iter:
            #print("Images in Batch: " + str(images))
            #print("Targets in Batch: " + str(targets))
            #print("\n\n\nT in targets:")
            images = list(image.to(device) for image in images)
        
            
            #print("Target index:", targets)    
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            #print("Targets list.", targets)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            print(losses.item())
            #print("This worked.")    
        lr_scheduler.step()
        #print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}')
        epoch_losses.append(losses.item())
        '''
    writer.flush()
    writer.close()
    '''
    torch.save(model.state_dict(), 'models/model_weights' + str(num_epochs) +'_'+str(batch_size)+'.pth')
    plt.figure(1)
    plt.plot(loss)
    plt.ylabel('Losses')
    plt.xlabel('Epochs')
    plt.title('Epochs Vs. Losses')
    plt.savefig('results/'+str(num_epochs)+'_'+str(batch_size)+'_losses.jpg')

    plt.figure(2)
    plt.plot(lr)
    plt.xlabel('Epochs')
    plt.ylabel('lr')
    plt.title('Epochs vs lr')
    plt.savefig('results/'+str(num_epochs) +'_'+str(batch_size)+'_lr.jpg')

    plt.figure(3)
    plt.plot(loss_classifier)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Classifier')
    plt.title('Epochs vs Loss Classifier')
    plt.savefig('results/'+str(num_epochs) +'_'+str(batch_size)+'_classifier.jpg')            


    plt.figure(4)
    plt.plot(loss_box_reg)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Box Reg')
    plt.title('Epochs vs Loss Box Reg')
    plt.savefig('results/'+str(num_epochs) +'_'+str(batch_size)+'_box_reg.jpg')

    plt.figure(5)
    plt.plot(loss_mask)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Mask')
    plt.title('Epochs vs Loss Mask')
    plt.savefig('results/'+str(num_epochs) +'_'+str(batch_size)+'_mask.jpg')

    plt.figure(6)
    plt.plot(loss_objectness)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Objectness')
    plt.title('Epochs vs Loss Objectness')
    plt.savefig('results/'+str(num_epochs) +'_'+str(batch_size)+'_objectness.jpg')

    plt.figure(7)
    plt.plot(loss_rpn_box_reg)
    plt.xlabel('Epochs')
    plt.ylabel('Loss RPN Box Reg')
    plt.title('Epochs vs Loss RPN')
    plt.savefig('results/'+str(num_epochs) +'_'+str(batch_size)+'_rpn_box_reg.jpg')
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--EPOCHS', default = 10)
    parser.add_argument('-b', '--BATCH', default = 1)
    parser.add_argument('-d', '--DATA', default = 'RRR')
    args = parser.parse_args()
    freeze_support()
    set_start_method('spawn')
    #p = Process(target = train(num_epochs ))
    p = Process(target =train(num_epochs = int(args.EPOCHS), batch_size = int(args.BATCH), data = str(args.DATA)))
    p.start()
    print("Finished.")
