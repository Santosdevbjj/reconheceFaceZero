## Criando um Sistema de Reconhecimento Facial do Zero.

![bairesDev](https://github.com/user-attachments/assets/19a85b48-9a12-43e0-b34f-df1210dd9148)



**Bootcamp BairesDev - Machine Learning Training** - **Ministrado pela DIO**

---

**DESCRIÇÃO**
O objetivo principal deste projeto é trabalhar com as bibliotecas e frameworks estudados e analisados em nossas aulas. Neste sentido, a proposta padrão envolve um sistema de detecção e reconhecimento de faces, utilizando o framework TensorFlow em conjuntos com as bibliotecas que o projetista julgue necessárias, de forma ilimitada.

---

# 🚀 reconheceFaceZero - Reconhecimento de faces 

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/) 
[![Docker](https://img.shields.io/badge/Docker-✓-2496ED.svg?logo=docker)](https://www.docker.com/) 
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-red)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU Support](https://img.shields.io/badge/CUDA-Enabled-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

---

## 📌 Descrição

Este projeto implementa um pipeline completo de **treinamento, inferência e prototipagem** com o **YOLOv8** para tarefas de detecção de objetos.  
O repositório foi estruturado para ser **modular, profissional e de fácil manutenção**, com suporte a:

- Treinamento customizado do YOLOv8  
- Inferência em imagens e vídeos  
- Execução em containers Docker (CPU/GPU com CUDA)  
- Prototipagem em **Jupyter Notebooks**  
- Testes automatizados para garantir qualidade do código  

---

## 📂 Estrutura do Projeto

### Visual (ASCII)

```
📦 reconheceFaceZero
├── 📜 Dockerfile           # Definição da imagem Docker para YOLOv8 + dependências
├── 📜 docker-compose.yml   # Orquestração de containers Docker (GPU/CPU)
├── 📜 Makefile             # Atalhos para build, run, train, test...
├── 📜 environment.yml      # Ambiente Conda para replicação local
├── 📜 requirements.txt     # Dependências Python para execução
├── ⚙️ setup.sh             # Configuração inicial do ambiente
├── 🚀 train.sh             # Script de treinamento do YOLOv8
├── 🔍 inference.sh         # Script de inferência em imagens/vídeos
├── 📓 run_jupyter.sh       # Inicializa Jupyter Notebook no container
│
├── 📁 notebooks/           # Notebooks Jupyter para prototipagem
│   └── 📓 exemplo.ipynb
│
├── 📁 src/                 # Código-fonte principal
│   └── 🛠️ utils.py
│
├── 📁 tests/               # Testes unitários e de integração
│   └── ✅ test_utils.py
│
└── 📜 README.md            # Documentação principal do projeto





---


Explicação dos principais arquivos/pastas

Dockerfile → Define o ambiente para rodar YOLOv8, Hugging Face e FastAPI.

docker-compose.yml → Orquestração de containers, com suporte a GPU (CUDA).

Makefile → Atalhos para make build, make train, make test.

environment.yml → Definição do ambiente Conda.

requirements.txt → Lista de bibliotecas Python.

setup.sh → Script de preparação inicial.

train.sh → Executa o treinamento do modelo YOLOv8.

inference.sh → Executa a inferência em imagens/vídeos.

run_jupyter.sh → Abre ambiente interativo em Jupyter Notebook.

notebooks/ → Contém experimentos e análises exploratórias.

src/ → Código-fonte (funções auxiliares, pré-processamento, etc).

tests/ → Testes automatizados para manter a qualidade do código.



---

🛠️ Tecnologias Utilizadas

Python 3.10+

YOLOv8 (Ultralytics)

PyTorch com suporte a CUDA

FastAPI para deploy da API

Docker + Docker Compose

Conda para gerenciamento de ambientes

Jupyter Notebook para prototipagem

pytest para testes automatizados



---

💻 Requisitos

Hardware

CPU (mínimo Quad-core)

GPU NVIDIA (recomendado para treino): suporte CUDA 11+

Memória RAM: mínimo 8 GB (16 GB recomendado)

Espaço em disco: 20 GB+ (datasets + modelos)


Software

Docker + NVIDIA Container Toolkit (para uso de GPU)

Python 3.10+ ou Conda

NVIDIA Drivers atualizados



---

🚀 Como Usar

1️⃣ Clonar o repositório

git clone https://github.com/seu-usuario/YOLOv8-Project.git
cd YOLOv8-Project

2️⃣ Criar ambiente Conda (opção local)

conda env create -f environment.yml
conda activate yolov8-env

3️⃣ Construir e rodar com Docker

make build
make run

4️⃣ Treinar modelo

./train.sh

5️⃣ Rodar inferência

./inference.sh


---

✅ Testes

Rodar testes automatizados:

pytest tests/


---

📜 Licença

Este projeto está sob a licença MIT.

---




---




