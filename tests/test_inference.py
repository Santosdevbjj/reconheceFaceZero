"""Teste simples para pipeline de inferência (requere modelos dummy no Drive)
"""
import os
from src.inference_pipeline import FaceRecognizer

DRIVE_ROOT = '/content/drive/MyDrive/reconheceFaceZero'
YOLO_MODEL = os.path.join(DRIVE_ROOT, 'models', 'yolo', 'yolov8_faces', 'weights', 'best.pt')
CLASS_MODEL = os.path.join(DRIVE_ROOT, 'models', 'classifier', 'svm_face_recog.pkl')


def test_pipeline_loads():
    fr = FaceRecognizer(YOLO_MODEL, CLASS_MODEL)
    assert fr is not None

if __name__ == '__main__':
    test_pipeline_loads()
    print('Teste básico OK')
