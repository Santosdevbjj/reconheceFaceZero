#!/usr/bin/env bash
# ============================================================
# Script de Inferência do Projeto reconheceFaceZero
# Autor: Sérgio Santos
# Objetivo: Rodar inferência em imagem ou vídeo com YOLOv8 + Classificador
# ============================================================

set -e  # interrompe se algum comando falhar

# Caminhos dos modelos
YOLO_MODEL="models/yolo/best.pt"
CLASSIFIER_MODEL="models/classifier/svm.pkl"

# Arquivo de entrada
INPUT=$1

if [ -z "$INPUT" ]; then
    echo "⚠️  Uso: ./inference.sh caminho/para/imagem_ou_video"
    exit 1
fi

echo "🚀 Rodando inferência no arquivo: $INPUT"

python3 - <<EOF
import cv2
from src.inference_pipeline import FaceRecognizer

# Caminhos dos modelos
yolo_path = "$YOLO_MODEL"
clf_path = "$CLASSIFIER_MODEL"

print("🔹 Carregando pipeline de reconhecimento facial...")
recog = FaceRecognizer(yolo_path, clf_path)

# Verifica se é vídeo ou imagem
if any("$INPUT".lower().endswith(ext) for ext in [".mp4", ".avi", ".mov"]):
    cap = cv2.VideoCapture("$INPUT")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = recog.recognize(frame)
        for (x1,y1,x2,y2,pred) in results:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, pred, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    frame = cv2.imread("$INPUT")
    results = recog.recognize(frame)
    for (x1,y1,x2,y2,pred) in results:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, pred, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.imshow("Resultado - Reconhecimento Facial", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
EOF

echo "✅ Inferência concluída!"
