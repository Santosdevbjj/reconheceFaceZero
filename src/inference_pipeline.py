"""
Pipeline único para inferência:
- carregar detector YOLO
- carregar classificador
- aceitar imagem BGR e retornar detections
"""
from ultralytics import YOLO
import joblib
import face_recognition

class FaceRecognizer:
    def __init__(self, yolo_weight_path, classifier_path):
        self.detector = YOLO(yolo_weight_path)
        self.clf = joblib.load(classifier_path)

    def recognize(self, image_bgr):
        results = self.detector(image_bgr)
        out = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                face = image_bgr[y1:y2, x1:x2]
                face_rgb = face[..., ::-1]
                encs = face_recognition.face_encodings(face_rgb)
                if len(encs) == 0:
                    continue
                pred = self.clf.predict([encs[0]])
                out.append({'box':(x1,y1,x2,y2), 'name':pred[0]})
        return out
