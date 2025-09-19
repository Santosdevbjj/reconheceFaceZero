#!/usr/bin/env bash
# ============================================================
# Script para rodar Jupyter Notebook no container Docker
# Projeto: reconheceFaceZero
# Autor: Sérgio Santos
# ============================================================

set -e  # interrompe execução em caso de erro

CONTAINER_NAME="reconhece-face-gpu"
PORT=8888

echo "🚀 Iniciando Jupyter Notebook no container $CONTAINER_NAME ..."

# Verifica se o container está rodando
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "🔹 Container já está em execução. Iniciando Jupyter..."
    docker exec -it $CONTAINER_NAME jupyter notebook --ip=0.0.0.0 --port=$PORT --no-browser --allow-root
else
    echo "⚠️  O container $CONTAINER_NAME não está rodando."
    echo "👉 Rode primeiro: make up-detach"
    exit 1
fi
