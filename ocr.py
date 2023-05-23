from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import torch
import numpy as np
from PIL import Image

from yolo_localizer import YOLOTextLocalizer
import torchvision.transforms as T

LOCALIZATION_WEIGHTS = './yolo_text_localization.pt'
OCR_WEIGHTS = './vgg_transformer.pth'
CONFIG_FILE = './base.yaml'
from Levenshtein import distance as lev_dist

def find_match_drugname(pres_name, lib_names):
    score = []
    for lib_name in lib_names:
        score.append(lev_dist(pres_name, lib_name))
    return lib_names[score.index(min(score))]

def find_pres_id_list(pres_names,mapping_id_name):
    lib_drugnames = [name for name in mapping_id_name['Drugname']]
    lib_drugnames = list(dict.fromkeys(lib_drugnames))
    pres_list, id_list = [], []
    for pres_name in pres_names:
        pres_lib_drugname = find_match_drugname(pres_name, lib_drugnames)
        df_id_pres_name = mapping_id_name[mapping_id_name['Drugname']==pres_lib_drugname]
        ids_pres = df_id_pres_name['classID'].to_list()
        pres_list = pres_list + len(ids_pres)*[pres_lib_drugname]
        id_list = id_list + ids_pres
    
    return pres_list, id_list 

def ID_drugname_match(yolo_ids, pres_names, mapping_id_name):
    pres_list, id_list = find_pres_id_list(pres_names,mapping_id_name)
    new_yolo_ids, yolo_drugnames = [], []
    for yolo_id in yolo_ids:
        if yolo_id not in id_list:
            new_yolo_ids.append(107)
            yolo_drugnames.append('Not_in_prescription')
        else:
            new_yolo_ids.append(yolo_id)
            yolo_drugnames.append(pres_list[id_list.index(yolo_id)])

    return  new_yolo_ids, yolo_drugnames
class TextDetector():

  def __init__(self, weight_path='./vgg_transformer.pth', localization_weights='',config_file='', imgsz=(640,640)):
    # self.config = Cfg.load_config_from_name(config_name)
    self.config = Cfg.load_config_from_file(config_file)
    self.config['weights'] = weight_path
    self.config['cnn']['pretrained']=True
    self.config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    self.predictor = Predictor(self.config)
    self.localizer = YOLOTextLocalizer(weights=localization_weights, imgsz=imgsz )
  def predict_locations(self, img_path,iou_thres=0.2, conf_thres=0.8):
     return self.localizer.predict(img_path,iou_thres=iou_thres, conf_thres=conf_thres)
  def predict_text_from_locations(self, localization_preds):
     # predicted_text = self.predictor.predict(img)
    diagnose_arr = []
    quantity_arr = []
    drugname_arr = []

    transform = T.ToPILImage()
    for pred in localization_preds:
      cut_im,ymin, conf, cls, box = pred
      cut_im = transform(cut_im)
      text = self.predictor.predict(cut_im)
      if cls.item() == 0:
        diagnose_arr.append([ymin,text])
      elif cls.item() == 1:
        drugname_arr.append([ymin,text])
      elif cls.item() == 2:
        quantity_arr.append([ymin,text])

    diagnose_arr = sorted(diagnose_arr, key=lambda x: x[0], reverse=False)
    drugname_arr = sorted(drugname_arr, key=lambda x: x[0], reverse=True)
    quantity_arr = sorted(quantity_arr, key=lambda x: x[0], reverse=True)

    diagnose_text = ''
    for diagnose in diagnose_arr:
      diagnose_text+= diagnose[1] + ' '

    num_drugname = min(len(drugname_arr), len(quantity_arr))
    drug_results = []
    for i in range(num_drugname):
      drug_results.append([drugname_arr[i][1],quantity_arr[i][1]])
    
    return diagnose_text, drug_results


if __name__ == '__main__':
  text_detector = TextDetector(weight_path=OCR_WEIGHTS,
                          localization_weights=LOCALIZATION_WEIGHTS, config_file=CONFIG_FILE)
  text_localization_preds = text_detector.predict_locations('./test_image/VAIPE_P_TRAIN_0.png')
  text_preds = text_detector.predict_text_from_locations(text_localization_preds)
  print(text_preds)