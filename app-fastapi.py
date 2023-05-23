import torch
import pandas as pd
from PIL import Image 
import io 
import shutil 
from pathlib import Path

from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import aiofiles
import uvicorn 

from ocr import TextDetector
from matching import extract_table_dict
app = FastAPI()

LOCALIZATION_WEIGHTS = './model_weight/yolo_text_localization.pt'
OCR_WEIGHTS = './model_weight/vgg_transformer.pth'
CONFIG_FILE = './model_weight/base.yaml'
YOLO_PATH = './model_weight/od_baseline.pt'
label_mapping_df = pd.read_csv('./label_mapping.csv')

# Yolo model for pill detection 
yolo_model = YOLO(YOLO_PATH)
# Text model for drugname extraction from prescription 
text_detector = TextDetector(weight_path=OCR_WEIGHTS,
                                     localization_weights=LOCALIZATION_WEIGHTS,
                                     config_file=CONFIG_FILE)

def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

def pill_processing(pill,yolo_model):  
    # file_bytes = pill.file.read()
    file_path = './test_image/'+ pill.filename
    save_upload_file(pill, Path(file_path)) 
    image = Image.open(file_path) #io.BytesIO(file_bytes)
    object_preds = yolo_model.predict(image)
    output_pill_dict = {'cls':object_preds[0].boxes.cls.to(torch.int32).tolist(),
                'boxes':object_preds[0].boxes.xywhn.tolist(),
                'confs':object_preds[0].boxes.conf.tolist(),
                }        
    return output_pill_dict

def pres_processing(pres, text_detector):
    pres_path = './test_image/'+ pres.filename
    save_upload_file(pres, Path(pres_path)) 
    text_localization_preds = text_detector.predict_locations(pres_path)
    text_preds = text_detector.predict_text_from_locations(text_localization_preds)
    text_localization_preds = [p[-1] for p in text_localization_preds]
    output_pres_dict = {'bboxes':text_localization_preds, 'text':list(text_preds)}
    return output_pres_dict

@app.get("/")
def home():
    return {"message": "VAIPE - MEDICAL PILL RECOGNITION"}

@app.post('/pill')
async def pill_process(file: UploadFile=File(...)):
    file_bytes = file.file.read()
    file_path = './test_image/'+ file.filename
    file_bytes.save(file_path)
    image = Image.open(file_path) #io.BytesIO(file_bytes)
    yolo_model = YOLO(YOLO_PATH)
    object_preds = yolo_model.predict(image)
    output_pill_dict = {'cls':object_preds[0].boxes.cls.to(torch.int32).tolist(),
                'boxes':object_preds[0].boxes.xywhn.tolist(),
                'confs':object_preds[0].boxes.conf.tolist(),
                }
    return output_pill_dict

@app.post('/pres')
async def pres_process(file: UploadFile):
    file_bytes = file.file.read()
    filename = file.filename
    pres_path = './test_image/'+ filename
    # file_bytes = file.file.read()
    # image = Image.open(io.BytesIO(file_bytes))
    text_detector = TextDetector(weight_path=OCR_WEIGHTS,
                                     localization_weights=LOCALIZATION_WEIGHTS,
                                     config_file=CONFIG_FILE)
    text_localization_preds = text_detector.predict_locations(pres_path)
    text_preds = text_detector.predict_text_from_locations(text_localization_preds)
    text_localization_preds = [p[-1] for p in text_localization_preds]
    output_pres_dict = {'bboxes':text_localization_preds, 'text':list(text_preds)}
    return output_pres_dict


@app.post("/main")
async def main(pres: UploadFile=File(...),pill: UploadFile=File(...)):
    output_pill_dict = pill_processing(pill,yolo_model)
    output_pres_dict = pres_processing(pres,text_detector)
    table,absence= extract_table_dict(output_pill_dict, output_pres_dict, label_mapping_df)
    return {"table": table,"absence": absence}

if __name__ == "__main__":
    uvicorn.run("app-fastapi:app",reload=True)