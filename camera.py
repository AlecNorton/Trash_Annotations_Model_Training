import pyrealsense2 as rs
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as transforms
import visualize
import test
from distinctipy import distinctipy

transform = transforms.Compose([
        transforms.ToTensor()])
#Load some variables
depth_max = 3
depth_min = .3
class_names = ['BG', 'Glass', 'Metal', 'Other', 'Paper', 'Plastic', 'Trash']


#Load function
def color_post_depth(pipeline, num_frames = 0):
    align = rs.align(rs.stream.color)
    spatial = rs.spatial_filter()
    #spatial.set_option(rs.option.holes_fill, 3)
    hole_filling = rs.hole_filling_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    if num_frames > 0:
        for x in range(num_frames):
            frameset = pipeline.wait_for_frames()
            frameset = align.process(frameset)
            frame = frameset.get_depth_frame()
            frame = depth_to_disparity.process(frame)
            frame = spatial.process(frame)
            frame = disparity_to_depth.process(frame)
            frame = hole_filling.process(frame)
    else:
        frameset = pipeline.wait_for_frames()
        frameset = align.process(frameset)
        frame = frameset.get_depth_frame()
        frame = depth_to_disparity.process(frame)
        frame = spatial.process(frame)
        frame = disparity_to_depth.process(frame)
        frame = hole_filling.process(frame)
    return frame, frameset.get_color_frame()





ctx = rs.context()
list = ctx.query_devices()
if(len(list) == 0):
    raise RuntimeError("No device connected.")
device = list[0]
print("Device, ", device)

#Pipeline
width, height = 1280, 720
cfg = rs.config()
cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
pipeline = rs.pipeline()
pipeline_profile = pipeline.start(cfg)

transform = transforms.Compose([
    transforms.ToTensor()
])

model = test.get_model_instance_segmentation(7)
model.load_state_dict(torch.load('models/model_weights10_10.pth', weights_only=True,  map_location=torch.device('cpu')))
model.eval()

class_names = ['BG', 'Glass', 'Metal', 'Other', 'Paper', 'Plastic', 'Trash']
colors = distinctipy.get_colors(len(class_names))


print(type(pipeline_profile))
cv2.namedWindow('Stream')
cv2.createTrackbar('Depth', 'Stream', 0, depth_max*1000, lambda x: x)
#Flag = 0, color stream, flag = 1, depth stream
flag = 0
mask_flag = 0
segm_flag = 0
try:
    while True:
        #Receive frames

        
        depth_frame, color_frame = color_post_depth(pipeline)        
        color = np.asanyarray(color_frame.get_data())

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_filter = cv2.getTrackbarPos('Depth', 'Stream')
        np_depth = np.asanyarray(depth_frame.get_data())
        if mask_flag:
            depth_mask = np.uint8(np.where(np_depth < depth_filter, 1, 0))
            color = cv2.bitwise_and(color, color, mask = depth_mask)
            colorized_depth = cv2.bitwise_and(colorized_depth, colorized_depth, mask = depth_mask)
        
        if segm_flag:
            color_tensor = transform(color)
            results = model([color_tensor])[0]
            boxes = np.int64(results['boxes'].detach().numpy())
            masks = results['masks'].detach().permute(0, 2, 3, 1)
            labels = np.int8(results['labels'].detach())
            scores = results['scores'].detach()

            color, combinedMask, color_masks, new_labels = visualize.visualize(color, masks, boxes, labels, class_names, scores, colors, 0, 100)

        cloud_data = []
        depth_profile = depth_frame.get_profile()
        depth_intrin = depth_profile.as_video_stream_profile().get_intrinsics()
        width = depth_intrin.width
        height = depth_intrin.height
        print("Width: ", width)
        print("Height: ", height)
    
        if (segm_flag or mask_flag):
            points_for_cloud = []
            for i in range(width):
                for j in range(height):
                    if color[j, i].any() == 0:
                        continue
                    else:
                        depth_pixel = [j, i]
                        depth = np_depth[j, i]
                        point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth / 1000)
                        



        
        cv2.circle(color, (1280//2, 720//2), (10), (255, 0, 0), -1)




        if flag == 0:
            cv2.imshow('Stream', color)
        elif flag == 1:
            cv2.imshow('Stream', colorized_depth)
        elif flag == 2:
            cv2.imshow('Stream', np.hstack((color, colorized_depth)))
        elif flag == 3:
            np_depth = np.asanyarray(depth_frame.get_data())
            depth_mask = np.uint8(np.where(np_depth < depth_filter, 1, 0))
            cv2.imshow('Stream', depth_mask*255)
                
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('d'):
            flag = 1
        elif key ==ord('c'):
            flag = 0
        elif key == ord('b'):
            flag = 2
        elif key == ord('m'):
            if mask_flag:
                mask_flag = False
            else:
                mask_flag = True
        elif key == ord('s'):
            if segm_flag:
                segm_flag = False
            else:
                segm_flag = True
        
finally:
    pipeline.stop()