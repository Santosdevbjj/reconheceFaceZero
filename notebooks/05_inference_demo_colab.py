# %%
"""Notebook 05 - Demo de Inferência (Colab)
- Carrega modelo YOLO (best.pt) e classificador
- Roda inferência em imagens e vídeo
"""
# %%
import os
import cv2
from ultralytics import YOLO
import joblib
import face_recognition

DRIVE_ROOT = '/content/drive/MyDrive/reconheceFaceZero'
YOLO_MODEL = os.path.join(DRIVE_ROOT, 'models', 'yolo', 'yolov8_faces', 'weights', 'best.pt')
CLASS_MODEL = os.path.join(DRIVE_ROOT, 'models', 'classifier', 'svm_face_recog.pkl')

# %%
# Carregar modelos
yolo = YOLO(YOLO_MODEL)
clf = joblib.load(CLASS_MODEL)

# %%
# Função utilitária para processar imagem e retornar detecções com nomes
def recognize_in_image(image_bgr):
    results = yolo(image_bgr)
    out = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = image_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(face_rgb)
            if len(encs) == 0:
                continue
            pred = clf.predict([encs[0]])
            out.append({'box':(x1,y1,x2,y2), 'name':pred[0]})
    return out

# %%
# Teste em uma imagem
img_path = os.path.join(DRIVE_ROOT, 'data', 'example.jpg')
if os.path.exists(img_path):
    img = cv2.imread(img_path)
    res = recognize_in_image(img)
    print(res)
else:
    print('Coloque uma imagem de teste em', img_path)
