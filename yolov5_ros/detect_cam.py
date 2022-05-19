# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run():
    weights=ROOT / 'yolov5s_person.pt'  # model.pt path(s)
    data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
    imgsz=(480, 640)  # inference size (height, width)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    dnn=False  # use OpenCV DNN for ONNX inference
    
    # Load model
    device = select_device('')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    
    # Get cam source
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while cv2.waitKey(33) < 0:
        ret, frame = capture.read()
        cv2.imshow("VideoFrame", frame)
        
        im = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im = np.reshape(im, ((1,) + im.shape))
        im = np.ascontiguousarray(im)
    
    # for path, im, im0s, vid_cap, s in dataset:
    #     print(type(im), im.shape)
    #     print(type(im0s), len(im0s), im0s)
    #     cv2.imshow('sibal', im0s[0])
        
        debug_t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        debug_t2 = time_sync()
        
        # Inference
        pred = model(im, augment=False, visualize=False)
        debug_t3 = time_sync()

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # batch_size >= 1
            #im0 = im0s[i].copy()
            im0 = frame
            
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    
            # Print time (inference-only)
            #LOGGER.info(f'{s}Done. ({debug_t3 - debug_t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            cv2.imshow('sibal result', im0)
            cv2.waitKey(1)  # 1 millisecond


if __name__ == "__main__":
    #check_requirements(exclude=('tensorboard', 'thop'))
    run()
