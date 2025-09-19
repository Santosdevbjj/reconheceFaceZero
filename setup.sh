#!/usr/bin/env bash
# ============================================================
# Script de Setup do Projeto reconheceFaceZero
# Autor: Sérgio Santos
# Objetivo: Automatizar criação de pastas, instalação e setup
# ============================================================

set -e  # para execução se algum comando falhar

PROJECT_NAME="reconheceFaceZero"

echo "🚀 Iniciando setup do projeto $PROJECT_NAME ..."

# ===========================
# Criar pastas
# ===========================
echo "📂 Criando estrutura de diretórios..."
mkdir -p data/persons
mkdir -p data/yolo_annotations/images/train
mkdir -p data/yolo_annotations/images/val
mkdir -p models/yolo
mkdir -p models/classifier
mkdir -p notebooks
mkdir -p src
mkdir -p config
mkdir -p tests

# ===========================
# Copiar arquivos base
# ===========================
echo "📝 Criando arquivos básicos se não existirem..."
touch requirements.txt
touch .gitignore
touch README.md
touch LICENSE

# ===========================
# Instalar dependências
# ===========================
echo "📦 Instalando dependências..."
if command -v pip >/dev/null 2>&1; then
    pip install -r requirements.txt
else
    echo "⚠️  pip não encontrado! Instale Python 3.9+ e pip antes de rodar este script."
fi

# ===========================
# Git Config
# ===========================
if [ -d .git ]; then
    echo "🔄 Repositório Git já inicializado."
else
    echo "📌 Inicializando repositório Git..."
    git init
    git branch -M main
    git add .
    git commit -m "🎉 Inicialização do projeto reconheceFaceZero"
fi

# ===========================
# Mensagem final
# ===========================
echo "✅ Setup concluído com sucesso!"
echo "👉 Agora você pode rodar: make build && make up-detach"
