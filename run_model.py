import torch
import pandas as pd

from ultralytics import YOLO
from ocr import TextDetector
from matching import extract_table_dict

# Weight paths
# LOCALIZATION_WEIGHTS = './model_weight/yolo_text_localization.pt'
# OCR_WEIGHTS = './model_weight/vgg_transformer.pth'
# CONFIG_FILE = './model_weight/base.yaml'
# YOLO_PATH = './model_weight/od_baseline.pt'

LOCALIZATION_WEIGHTS = './model_weight/yolo_text_localization.pt'
OCR_WEIGHTS = './model_weight/vgg_transformer.pth'
CONFIG_FILE = './model_weight/base.yaml'
YOLO_PATH = './model_weight/od_baseline.pt'

# Image path 
pres_path = './test_image/VAIPE_P_TRAIN_0.png'
pill_path = './test_image/VAIPE_P_0_0.jpg'

# Import ID_drugname_mapping file
label_mapping_df = pd.read_csv('./label_mapping.csv')

#process pill image 
yolo_model = YOLO(YOLO_PATH)
object_preds = yolo_model.predict(pill_path)
output_pill_dict = {'cls':object_preds[0].boxes.cls.to(torch.int32).tolist(),
                  'boxes':object_preds[0].boxes.xywhn.tolist(),
                  'confs':object_preds[0].boxes.conf.tolist(),
                  }

# process pres image
text_detector = TextDetector(weight_path=OCR_WEIGHTS,
                                     localization_weights=LOCALIZATION_WEIGHTS,
                                     config_file=CONFIG_FILE)

text_localization_preds = text_detector.predict_locations(pres_path)
text_preds = text_detector.predict_text_from_locations(text_localization_preds)
text_localization_preds = [p[-1] for p in text_localization_preds]
output_pres_dict = {'bboxes':text_localization_preds, 'text':list(text_preds)}

# match the result of OCR and YOLO 
table,absence= extract_table_dict(output_pill_dict, output_pres_dict, label_mapping_df)
print(table)
print(absence)