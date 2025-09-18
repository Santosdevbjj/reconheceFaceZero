"""
Wrapper simples para treinar YOLOv8 via API
"""
from ultralytics import YOLO
import os


def train_yolo(data_yaml, output_dir, epochs=50, imgsz=640, batch=16, model='yolov8n.pt'):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, project=output_dir, name='yolov8_faces')
    # retorna caminho provável do best.pt
    best = os.path.join(output_dir, 'yolov8_faces', 'weights', 'best.pt')
    return best
