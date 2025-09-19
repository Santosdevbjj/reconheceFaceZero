#!/usr/bin/env bash
# ============================================================
# Script de Treinamento do Projeto reconheceFaceZero
# Autor: Sérgio Santos
# Objetivo: Automatizar o treinamento do YOLOv8 e do classificador de faces
# ============================================================

set -e  # interrompe em caso de erro

YOLO_DATA="config/data.yaml"
YOLO_MODEL="models/yolo/best.pt"
CLASSIFIER_MODEL="models/classifier/svm.pkl"

echo "🚀 Iniciando pipeline de treinamento..."

# ===========================
# Treinamento YOLOv8
# ===========================
echo "📦 Treinando YOLOv8..."
python3 - <<EOF
from ultralytics import YOLO

print("🔹 Carregando YOLOv8n pré-treinado...")
model = YOLO("yolov8n.pt")

print("🔹 Iniciando treinamento...")
results = model.train(data="$YOLO_DATA", epochs=50, imgsz=640)

print("💾 Salvando melhor modelo em $YOLO_MODEL")
model.save("$YOLO_MODEL")
EOF

# ===========================
# Extração de embeddings
# ===========================
echo "📦 Extraindo embeddings..."
python3 src/embeddings.py

# ===========================
# Treinamento do Classificador
# ===========================
echo "📦 Treinando classificador de faces..."
python3 - <<EOF
import numpy as np
import joblib
from sklearn.svm import SVC
from pathlib import Path

# Carregar embeddings e labels
X = np.load("models/embeddings.npy")
y = np.load("models/labels.npy")

print("🔹 Treinando SVM com kernel linear...")
clf = SVC(kernel="linear", probability=True)
clf.fit(X, y)

Path("models/classifier").mkdir(parents=True, exist_ok=True)
joblib.dump(clf, "$CLASSIFIER_MODEL")
print(f"💾 Classificador salvo em {CLASSIFIER_MODEL}")
EOF

echo "✅ Pipeline de treinamento concluído com sucesso!"
