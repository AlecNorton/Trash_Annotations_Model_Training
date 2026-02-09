
import distinctipy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def convert_masks(masks, labels, colors, size):
    color_masks = []
    new_labels = []
    combinedMask = None
    for i in range(len(labels)):
        color = np.full((720, 1280, 3), colors[labels[i]])
        mask = torch.round(masks[i])
        mask = np.uint8(mask)
        if (np.count_nonzero(mask) < size):
            continue
        color_masks.append(cv2.bitwise_and(color, color, mask = mask))
        new_labels.append(labels[i])

        if combinedMask is None:
            combinedMask = color_masks[-1]
        else:
            combinedMask = combinedMask + color_masks[-1]
    return (combinedMask), color_masks, new_labels

def visualize(image, masks, boxes, labels, class_names, scores, colors, score_threshold = .5, size_threshold = 100):
    #print("COLORS", colors)
    idx = np.where(scores >= score_threshold)
    masks = masks[idx]
    boxes = boxes[idx]
    scores = scores[idx]
    labels = labels[idx]
    combinedMask, color_masks, new_labels = convert_masks(masks, labels, colors, size_threshold)
    #print("COLORS", colors)
    combinedMask = np.uint8(combinedMask*255)
    image = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), .5, combinedMask, .5, 0)
    for i in range(len(idx[0])):
        #print(i)
        color = tuple(i*255 for i in colors[labels[i]])
        cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), color, 2)
        cv2.putText(image, str(class_names[labels[i]]), (boxes[i][0], boxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    plt.imshow(image)
    plt.show()
    cv2.waitKey(0)  
