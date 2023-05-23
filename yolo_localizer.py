# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]/'yolov5'  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (Profile, check_file, check_img_size, check_imshow,
                           increment_path, non_max_suppression)
from yolov5.utils.torch_utils import select_device, smart_inference_mode

def xyxy_to_yolo(x1, y1, x2, y2, img_width, img_height):
    # Compute the center coordinates of the bounding box
    bbox_center_x = (x1 + x2) / 2.0
    bbox_center_y = (y1 + y2) / 2.0

    # Compute the width and height of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Compute the YOLO coordinates of the bounding box
    yolo_x = bbox_center_x / img_width
    yolo_y = bbox_center_y / img_height
    yolo_width = bbox_width / img_width
    yolo_height = bbox_height / img_height
    
    return yolo_x, yolo_y, yolo_width, yolo_height

class YOLOTextLocalizer:

    @smart_inference_mode()
    def __init__(self, weights, imgsz=(640,640)):
        # Load model
        # device = select_device('')
        device = torch.device('cpu')
        model = DetectMultiBackend(weights,
                                   device=device,
                                   dnn=False,
                                   data=ROOT / 'data/coco128.yaml',
                                   fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.imgsz = imgsz
        self.model = model
    
    @smart_inference_mode()
    def predict(self, source ,conf_thres=0.25,iou_thres=0.45,max_det=1000):
        source = str(source)
        bs = 1  # batch_size
        screenshot = source.lower().startswith('screen')
        if screenshot:
            dataset = LoadScreenshots(source, img_size=self.imgsz, stride=self.model.stride, auto=self.model.pt)
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.model.stride, auto=self.model.pt, vid_stride=1)
        
        self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            # with dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            # with dt[1]:
            pred = self.model(im, augment=False, visualize=False)

            # NMS
            # with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
            
            # Get cut images
            # pred[0][:,:4] =pred[0][:,:4].to(torch.uint8)

            im = im*255
            im = im.to(torch.uint8)
            im = im[0]
            results =[ ]
            for box in list(pred[0]):
                x,y,x1,y1 = box[:4].to(torch.int32)
                x,y,x1,y1 = x.item(),y.item(),x1.item(),y1.item()
                conf = box[4]
                cls = box[5].to(torch.int32)
                cut_im = im[:,y:y1,x:x1]
                # print(cut_im.shape)
                x,y,w,h = xyxy_to_yolo(x,y,x1,y1,img_height=im.shape[1], img_width=im.shape[2])
                results.append([cut_im,y,conf,cls, [x,y,w,h]])
        return results

LOCALIZATION_WEIGHTS = './yolo_text_localization.pt'
if __name__ == '__main__':
    localizer = YOLOTextLocalizer(weights=LOCALIZATION_WEIGHTS)
    print(localizer.predict(source='/VAIPE_P_TRAIN_0.png'),)