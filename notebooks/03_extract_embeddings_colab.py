# %%
"""Notebook 03 - Extrair embeddings com face_recognition
- Percorre `data/persons` e extrai face_encodings por imagem
- Salva arrays em .npy para treino do classificador
"""
# %%
import os
import numpy as np
import face_recognition
from tqdm import tqdm

DRIVE_ROOT = '/content/drive/MyDrive/reconheceFaceZero'
PERSONS_DIR = os.path.join(DRIVE_ROOT, 'data', 'persons')
EMB_PATH = os.path.join(DRIVE_ROOT, 'models', 'embeddings.npy')
LBL_PATH = os.path.join(DRIVE_ROOT, 'models', 'labels.npy')

embs = []
labels = []
for person in os.listdir(PERSONS_DIR):
    ppath = os.path.join(PERSONS_DIR, person)
    if not os.path.isdir(ppath):
        continue
    for f in os.listdir(ppath):
        if not f.lower().endswith(('jpg','png','jpeg')):
            continue
        img_path = os.path.join(ppath, f)
        image = face_recognition.load_image_file(img_path)
        enc = face_recognition.face_encodings(image)
        if len(enc) == 0:
            print('Aviso: não foi possível extrair face de', img_path)
            continue
        embs.append(enc[0])
        labels.append(person)

embs = np.array(embs)
labels = np.array(labels)

np.save(EMB_PATH, embs)
np.save(LBL_PATH, labels)
print('Embeddings e labels salvos em', EMB_PATH, LBL_PATH)
