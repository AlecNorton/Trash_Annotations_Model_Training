print("Begun")
import sys
import torchvision
import torchvision.transforms as transforms
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from pycocotools import mask as maskUtils
import numpy as np
import argparse
from multiprocessing import Process, freeze_support, set_start_method
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

INTEL_SIZE = (1280, 720)

def train(num_epochs, batch_size):
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
            masks = torch.tensor(np.array([mask]), dtype = torch.uint8)
            ret.append({'boxes': boxes, 'labels': labels, 'masks': masks})
        return ret
    print("Loading dataset.")
    dataset = torchvision.datasets.CocoDetection('data', 'data/resized_17/annotations.json', transform = transform, target_transform= target_transform)
    print("Dataset load complete.")
    #Split dataset.
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    #Create data loader.
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = lambda batch: tuple(zip(*batch)))
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, collate_fn = lambda batch: tuple(zip(*batch)))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = lambda batch: tuple(zip(*batch)))
    #Initialize a Mask R-CNN model with pretrained weights. 
    
    model = maskrcnn_resnet50_fpn_v2(weights = 'DEFAULT')
    
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels = in_features_box, num_classes = 7)

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels = in_features_mask, dim_reduced = dim_reduced, num_classes = 7)

    #Load parameters and optimizer. 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #plt.imshow(imageTransform(image))
    #plt.imshow(imageTransform(target['masks']), alpha = 0.5)
    #plt.show()
    #cv2.waitKey()
    print("EPOCHS: " + str(num_epochs))
    print("BATCH_SIZE: " + str(batch_size))
    epoch_losses = []
    
    for epoch in range(num_epochs):
        print("EPOCH: ", epoch)
        for images, targets in data_loader:
            #print("Images in Batch: " + str(images))
            #print("Targets in Batch: " + str(targets))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t[0].items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            print(losses.item())
                
        lr_scheduler.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}')
        epoch_losses.append(losses.item())
    
    torch.save(model.state_dict(), 'models/model_weights_17_' + str(num_epochs) +'_'+str(batch_size)+'.pth')
    plt.figure(1)
    plt.plot(epoch_losses)
    plt.ylabel('Losses')
    plt.xlabel('Epochs')
    plt.title('Epochs Vs. Losses')
    plt.savefig('results/'+str(num_epochs)+'_'+str(batch_size)+'_losses.jpg')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--EPOCHS', default = 10)
    parser.add_argument('-b', '--BATCH', default = 5)
    args = parser.parse_args()
    freeze_support()
    set_start_method('spawn')
    p = Process(target =train(num_epochs = int(args.EPOCHS), batch_size = int(args.BATCH)))
    p.start()
