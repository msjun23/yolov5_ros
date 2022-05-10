#!/usr/bin/env python3

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import gi
gi.require_version('Gtk', '2.0')

import rospy
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from yolov5_ros.msg import BoundingBox, BoundingBoxes

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

class Detector:
    def __init__(self):
        # Get parameters
        self.image_topic = rospy.get_param('~image_topic')      # image topic to subscribe
        self.weights = rospy.get_param('~weights')              # weight model path(s)
        self.data = rospy.get_param('~data')                    # dataset.yaml path
        self.conf_thres = rospy.get_param('~conf_thres', 0.25)  # confidence threshold
        self.w = rospy.get_param('~width', 640)                 # Width
        self.h = rospy.get_param('~height', 480)                # Height
        self.imgsz = (self.h, self.w)                           # inference size (height, width)

        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        
        # Load model
        rospy.loginfo("Loading model...")
        rospy.loginfo(self.weights)
        self.device = select_device('') # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        if 'engine' in self.weights:    # If using TensorRT
            imgsz = (640, 640)
        else:                           # Using .pt
            imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        
        # Half
        self.half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()
            
        # Dataloader
        cudnn.benchmark = True  # set True to speed up constant image size inference
    
        # Run inference
        self.model.warmup(imgsz=(1, 3, *imgsz), half=self.half)  # warmup
        
        rospy.Subscriber(self.image_topic, Image, self.Detector)
        self.pub_detected_img = rospy.Publisher('/detected_img', Image, queue_size=1)
        self.pub_bbxes = rospy.Publisher('/bounding_box_array', BoundingBoxes, queue_size=1)
        
        rospy.spin()

    def Detector(self, img_data):
        bridge = CvBridge()
        try:
            cv_img = bridge.imgmsg_to_cv2(img_data, desired_encoding='bgr8')
            if 'engine' in self.weights:    # If using TensorRT
                cv_img = cv2.resize(cv_img, dsize=(640, 640), interpolation=cv2.INTER_AREA)
            else:
                cv_img = cv2.resize(cv_img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        except CvBridgeError as e:
            print(e)
        
        # By numpy method
        #cv_img = np.frombuffer(img_data.data, dtype=np.uint8).reshape(img_data.height, img_data.width, -1)
        
        # Image test
        #cv2.imshow("D435i Image", cv_img)
        #cv2.waitKey(1)
        
        # Incference with cv_img
        im = cv_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        im = np.reshape(im, ((1,) + im.shape))
        im = np.ascontiguousarray(im)
        
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
            
        # Inference
        pred = self.model(im, augment=False, visualize=False)
        
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            # det: [[xmin, ymin, xmax, ymax, probability, label],
            #       [xmin, ymin, xmax, ymax, probability, label], ...]
            
            # batch_size >= 1
            im0 = cv_img
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                cnt = 0
                bbx_arr = BoundingBoxes()
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
                    # BoundingBox ROS topic
                    bbx = BoundingBox()
                    bbx.Class = self.names[c]
                    bbx.probability = float(f'{conf:.2f}')
                    if 'engine' in self.weights:    # If using TensorRT
                        # 640x640 -> 640x480
                        bbx.xmin = int(xyxy[0].item())
                        bbx.ymin = int(480/640 * xyxy[1].item())
                        bbx.xmax = int(xyxy[2].item())
                        bbx.ymax = int(480/640 * xyxy[3].item())
                    else:
                        bbx.xmin = int(xyxy[0].item())
                        bbx.ymin = int(xyxy[1].item())
                        bbx.xmax = int(xyxy[2].item())
                        bbx.ymax = int(xyxy[3].item())
                    
                    # BoundingBoxes ROS topic
                    bbx_arr.header = img_data.header
                    bbx_arr.bounding_boxes.append(bbx)
                    
                    cnt = cnt + 1
                
                # Publish BoundingBoxes topic
                self.pub_bbxes.publish(bbx_arr)
                rospy.loginfo("Detected %d objects", cnt)

            # Stream results
            im0 = annotator.result()
            # im0 = cv2.resize(im0, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            self.pub_detected_img.publish(bridge.cv2_to_imgmsg(im0, encoding="bgr8"))
            cv2.imshow('result', im0)
            cv2.waitKey(1)  # 1 millisecond


if __name__ == "__main__":
    rospy.init_node('DoorDetector')
    
    detector = Detector()
    
