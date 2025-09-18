"""
Detecta faces com YOLO e salva crops em pasta destino
"""
from ultralytics import YOLO
import cv2
import os


def detect_and_save_crops(yolo_weight, image_paths, out_dir, min_area=1000):
    os.makedirs(out_dir, exist_ok=True)
    yolo = YOLO(yolo_weight)
    idx = 0
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        res = yolo(img)
        for r in res:
            boxes = r.boxes
            for b in boxes:
                x1,y1,x2,y2 = map(int,b.xyxy[0])
                w,h = x2-x1, y2-y1
                if w*h < min_area:
                    continue
                crop = img[y1:y2, x1:x2]
                fname = os.path.join(out_dir, f'crop_{idx:06d}.jpg')
                cv2.imwrite(fname, crop)
                idx += 1
    return idx
