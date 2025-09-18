"""
Funções para extrair embeddings de faces
- usa face_recognition por padrão
- opção de usar modelo TF (FaceNet) se disponível
"""
import face_recognition
import numpy as np


def extract_encoding_from_image_path(img_path):
    img = face_recognition.load_image_file(img_path)
    enc = face_recognition.face_encodings(img)
    if len(enc) == 0:
        return None
    return enc[0]


def extract_encoding_from_bgr(bgr_image):
    # converte para RGB
    rgb = bgr_image[..., ::-1]
    enc = face_recognition.face_encodings(rgb)
    if len(enc) == 0:
        return None
    return enc[0]
