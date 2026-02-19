
import distinctipy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def iou_score(mask1, mask2):
    #plt.figure(1)
    #plt.title("New Mask")
    #plt.imshow(mask1)
    #plt.figure(2)
    #plt.title("Old masks.")
    #plt.imshow(mask2)
    #plt.show()
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou_score = intersection / (union + 1e-6) #So we don't divide by zero.
    return iou_score 

def convert_masks(masks, labels, colors, boxes, size):
    color_masks = []
    new_labels = []
    new_masks = []
    new_boxes = []
    combinedMask = None
    for i in range(len(labels)):
        color = np.full((720, 1280, 3), colors[labels[i]])
        mask = torch.round(masks[i])
        mask = np.uint8(mask)
        if (np.count_nonzero(mask) < size):
            continue
        copyFlag = 0
        for j in range(len(new_masks)):
            print(j)
            score = iou_score(mask, new_masks[j])
            if score > .9:
                copyFlag = 1
                break
        if copyFlag ==1:
            continue
                
        color_masks.append(cv2.bitwise_and(color, color, mask = mask))
        new_labels.append(labels[i])
        new_masks.append(mask)

        new_boxes.append(boxes[i])


        if combinedMask is None:
            combinedMask = color_masks[-1]
        else:
            combinedMask = combinedMask + color_masks[-1]
    return (combinedMask), color_masks, new_labels, new_boxes

def visualize(image, masks, boxes, labels, class_names, scores, colors, score_threshold = .5, size_threshold = 100):
    #print("COLORS", colors)
    idx = np.where(scores >= score_threshold)
    masks = masks[idx]
    boxes = boxes[idx]
    scores = scores[idx]
    labels = labels[idx]
    combinedMask, color_masks, new_labels, new_boxes = convert_masks(masks, labels, colors, boxes, size_threshold)
    combinedMask = np.uint8(combinedMask*255)
    image = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), .5, combinedMask, .5, 0)
    for i in range(len(new_labels)):
        color = tuple(i*255 for i in colors[new_labels[i]])
        cv2.rectangle(image, (new_boxes[i][0], new_boxes[i][1]), (new_boxes[i][2], new_boxes[i][3]), color, 2)
        cv2.putText(image, str(class_names[new_labels[i]]), (new_boxes[i][0], new_boxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image, color_masks, new_labels
