# %%
"""Notebook 02 - Treinar YOLOv8 (Colab)
- Carrega ultralytics
- Treina modelo a partir do data.yaml
"""
# %%
from ultralytics import YOLO
import os

DRIVE_ROOT = '/content/drive/MyDrive/reconheceFaceZero'
DATA_YAML = os.path.join(DRIVE_ROOT, 'config', 'data.yaml')
MODEL_OUTPUT = os.path.join(DRIVE_ROOT, 'models', 'yolo')
os.makedirs(MODEL_OUTPUT, exist_ok=True)

# %%
# Carregar modelo pré-treinado pequeno
model = YOLO('yolov8n.pt')

# %%
# Treinar (ajuste epochs conforme dataset)
model.train(data=DATA_YAML, epochs=50, imgsz=640, batch=16, device=0, project=MODEL_OUTPUT, name='yolov8_faces')

# %%
# Após treino, salvar melhor.pt para uso
# A API do ultralytics salva automaticamente no diretório project/name/runs/train/expX/weights/best.pt
print('Verifique o diretório de saída para encontrar o melhor checkpoint')
