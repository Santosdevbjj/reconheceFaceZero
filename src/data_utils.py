"""
Utilitários para manipulação do dataset
- converter anotações
- checar imagens sem faces
- balancear dataset
"""
import os
from PIL import Image


def list_person_images(root_dir):
    """Retorna dicionário {person: [paths]}"""
    out = {}
    for person in os.listdir(root_dir):
        p = os.path.join(root_dir, person)
        if not os.path.isdir(p):
            continue
        imgs = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('jpg','png','jpeg'))]
        out[person] = imgs
    return out


def check_image_sizes(root_dir, max_side=2000):
    """Checa imagens muito grandes e retorna a lista"""
    big = []
    for _, imgs in list_person_images(root_dir).items():
        for im in imgs:
            try:
                with Image.open(im) as I:
                    if max(I.size) > max_side:
                        big.append(im)
            except Exception as e:
                print('Erro ao abrir', im, e)
    return big
