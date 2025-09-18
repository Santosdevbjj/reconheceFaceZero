# %%
"""
Notebook 01 - Preparar ambiente e dados (Colab)
Conteúdo: monta Drive, instala dependências, checa estrutura de pastas e visualiza imagens.
"""
# %%
# Instalações (executar apenas em Colab)
!pip install --upgrade pip
!pip install ultralytics opencv-python-headless face_recognition scikit-learn matplotlib tqdm joblib tensorflow

# %%
# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# %%
# Defina caminho base
DRIVE_ROOT = '/content/drive/MyDrive/reconheceFaceZero'

import os
os.makedirs(DRIVE_ROOT, exist_ok=True)

# %%
# Estrutura sugerida de pastas (criando se não existir)
folders = [
    f'{DRIVE_ROOT}/data/persons',
    f'{DRIVE_ROOT}/data/yolo_annotations/images/train',
    f'{DRIVE_ROOT}/data/yolo_annotations/images/val',
    f'{DRIVE_ROOT}/data/yolo_annotations/labels/train',
    f'{DRIVE_ROOT}/data/yolo_annotations/labels/val',
    f'{DRIVE_ROOT}/models/yolo',
    f'{DRIVE_ROOT}/models/classifier'
]
for f in folders:
    os.makedirs(f, exist_ok=True)
print('Pastas criadas/verificadas:')
for f in folders:
    print('-', f)

# %%
# Ferramenta rápida para visualizar algumas imagens
from IPython.display import display
from PIL import Image
sample_dir = f'{DRIVE_ROOT}/data/persons'
# se houver imagens, mostra as primeiras
if os.path.exists(sample_dir):
    for person in os.listdir(sample_dir)[:3]:
        p = os.path.join(sample_dir, person)
        if os.path.isdir(p):
            imgs = [x for x in os.listdir(p) if x.lower().endswith(('jpg','png','jpeg'))][:3]
            for im in imgs:
                display(Image.open(os.path.join(p, im)).resize((300,300)))
else:
    print('Nenhuma imagem encontrada em', sample_dir)
